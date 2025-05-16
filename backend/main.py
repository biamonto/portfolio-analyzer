from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import httpx
from collections import defaultdict
from typing import Optional
import numpy as np
import statsmodels.api as sm



app = FastAPI()

class Asset(BaseModel):
    symbol: str
    weight: float

class Portfolio(BaseModel):
    assets: List[Asset]
    start: str  # datum i format "YYYY-MM-DD"
    end: str

class ExtendedPortfolio(Portfolio):
    horizon: Optional[int] = 1  # Standard är 1-dag

class ETFRequest(BaseModel):
    symbol: str
    start: str  # YYYY-MM-DD
    end: str

class ETFAlphaRequest(BaseModel):
    symbol: str
    benchmark: str = "SPY"
    start: str
    end: str

class ETFFactorRequest(BaseModel):
    symbol: str
    start: str
    end: str

@app.get("/")
def read_root():
    return {"message": "Din FastAPI-backend körs!"}

@app.post("/analyze")
def analyze_portfolio(portfolio: Portfolio):
    symbols = [a.symbol for a in portfolio.assets]
    weights_dict = {a.symbol: a.weight for a in portfolio.assets}

    raw = yf.download(symbols, start=portfolio.start, end=portfolio.end)
    print(raw)
    if raw.empty:
        return {"error": "Ingen data hittades för valda symboler och datum."}

    # Om vi har flera aktier kommer raw ha MultiIndex kolumner
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            df = raw['Close']
        except KeyError:
            return {"error": "'Close' finns inte i hämtad data."}
    else:
        # Bara en aktie
        try:
            df = raw[['Close']]
            df.columns = [symbols[0]]
        except KeyError:
            return {"error": "'Close' finns inte i hämtad data."}
    df = df.dropna()

    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    ef = EfficientFrontier(mu, S)
    ef.set_weights(weights_dict) 
    perf = ef.portfolio_performance(verbose=False)

    return {
        "expected_return": round(perf[0], 4),
        "volatility": round(perf[1], 4),
        "sharpe_ratio": round(perf[2], 2),
    }

@app.post("/optimize")
def optimize_portfolio(portfolio: Portfolio):
    symbols = [a.symbol for a in portfolio.assets]
    df = yf.download(symbols, start=portfolio.start, end=portfolio.end)["Close"]
    df = df.dropna()

    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned = ef.clean_weights()

    return cleaned

@app.post("/history")
def get_portfolio_history(portfolio: Portfolio):
    import pandas as pd

    symbols = [a.symbol for a in portfolio.assets]
    weights_dict = {a.symbol: a.weight for a in portfolio.assets}

    raw = yf.download(symbols, start=portfolio.start, end=portfolio.end)
    if raw.empty:
        return {"error": "Ingen data hittades."}

    if isinstance(raw.columns, pd.MultiIndex):
        try:
            df = raw["Close"]
        except KeyError:
            return {"error": "'Close' saknas i data."}
    else:
        try:
            df = raw[["Close"]]
            df.columns = [symbols[0]]
        except KeyError:
            return {"error": "'Close' saknas i data."}

    df = df.dropna()

    # Normalisera varje tillgångs pris till 1.0 i början
    df_normalized = df / df.iloc[0]

    # Vikta enligt användarens input
    for col in df_normalized.columns:
        df_normalized[col] *= weights_dict.get(col, 0)

    # Summera till portföljens dagliga värde
    df_normalized["Total"] = df_normalized.sum(axis=1)

    return {
        "dates": df_normalized.index.strftime("%Y-%m-%d").tolist(),
        "values": df_normalized["Total"].round(4).tolist()
    }

@app.post("/portfolio-characteristics")
def get_portfolio_characteristics(portfolio: Portfolio):
    from collections import defaultdict
    import httpx

    FINNHUB_API_KEY = "d0h5cu9r01qv1u35cv5gd0h5cu9r01qv1u35cv60"

    symbols = [a.symbol.upper() for a in portfolio.assets]
    weights = {a.symbol.upper(): a.weight for a in portfolio.assets}

    currency_weights = defaultdict(float)
    sector_weights = defaultdict(float)
    region_weights = defaultdict(float)

    with httpx.Client() as client:
        for symbol in symbols:
            try:
                r = client.get(
                    "https://finnhub.io/api/v1/stock/profile2",
                    params={"symbol": symbol, "token": FINNHUB_API_KEY},
                    timeout=5
                )
                data = r.json()
                w = weights[symbol]
                currency = data.get("currency", "UNKNOWN")
                sector = data.get("finnhubIndustry", "UNKNOWN")
                region = data.get("country", "UNKNOWN")

                currency_weights[currency] += w
                sector_weights[sector] += w
                region_weights[region] += w

            except Exception:
                currency_weights["UNKNOWN"] += weights[symbol]
                sector_weights["UNKNOWN"] += weights[symbol]
                region_weights["UNKNOWN"] += weights[symbol]

    return {
        "currency_weights": dict(currency_weights),
        "sector_weights": dict(sector_weights),
        "region_weights": dict(region_weights)
    }

@app.post("/var")
def calculate_var(portfolio: ExtendedPortfolio):
    import pandas as pd
    import numpy as np

    symbols = [a.symbol for a in portfolio.assets]
    weights = {a.symbol: a.weight for a in portfolio.assets}
    horizon = max(1, portfolio.horizon or 1)

    raw = yf.download(symbols, start=portfolio.start, end=portfolio.end)
    if raw.empty:
        return {"error": "Ingen data hittades."}

    if isinstance(raw.columns, pd.MultiIndex):
        try:
            df = raw["Close"]
        except KeyError:
            return {"error": "'Close' saknas i data."}
    else:
        try:
            df = raw[["Close"]]
            df.columns = [symbols[0]]
        except KeyError:
            return {"error": "'Close' saknas i data."}

    df = df.dropna()
    returns = df.pct_change().dropna()

    for col in returns.columns:
        returns[col] *= weights.get(col, 0)

    portfolio_returns = returns.sum(axis=1)

    # Aggregera över önskad horisont
    if horizon > 1:
        portfolio_returns = portfolio_returns.rolling(horizon).sum().dropna()

    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

    return {
        "VaR_95": round(var_95, 4),
        "CVaR_95": round(cvar_95, 4),
        "mean_return": round(portfolio_returns.mean(), 4),
        "std_dev": round(portfolio_returns.std(), 4),
        "n_obs": len(portfolio_returns),
        "returns": portfolio_returns.round(4).tolist(),
        "var_threshold": round(var_95, 4)
    }

@app.post("/etf-analysis")
def analyze_etf(data: ETFRequest):
    
    try:
        df = yf.download(data.symbol, start=data.start, end=data.end, progress=False)

        # Om det är MultiIndex: välj rätt nivå
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"].dropna()
        else:
            df = df["Close"].dropna()

        df = pd.DataFrame(df).rename(columns={data.symbol: "Price", "Close": "Price"})

        
        returns = df["Price"].pct_change().dropna()

        # Beräkningar
        total_return = (df["Price"][-1] / df["Price"][0]) - 1
        days = (pd.to_datetime(data.end) - pd.to_datetime(data.start)).days
        cagr = (1 + total_return) ** (365 / days) - 1
        std_dev = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        downside = returns[returns < 0]
        sortino = returns.mean() / downside.std() * np.sqrt(252) if not downside.empty else None

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1
        max_dd = drawdown.min()

        # % positiva månader
        monthly = df["Price"].resample("M").last().pct_change().dropna()
        pos_months = (monthly > 0).sum() / len(monthly)

        def safe_round(x, decimals=4):
            if pd.isna(x) or np.isinf(x):
                return None
            return round(x, decimals)

        
        return {
            "symbol": data.symbol.upper(),
            "total_return": safe_round(total_return),
            "cagr": safe_round(cagr),
            "sharpe": safe_round(sharpe, 2),
            "sortino": safe_round(sortino, 2) if sortino else None,
            "std_dev": safe_round(std_dev),
            "max_drawdown": safe_round(max_dd),
            "positive_months": safe_round(pos_months, 4),
            "history": df["Price"].round(2).to_dict()
        }


    except Exception as e:
        return {"error": str(e)}

@app.post("/etf-alpha")
def etf_alpha_analysis(req: ETFAlphaRequest):

    try:
        # Ladda ETF och benchmark
        df = yf.download([req.symbol, req.benchmark], start=req.start, end=req.end, progress=False)

        # Hantera MultiIndex korrekt
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" not in df.columns.levels[0]:
                print(df)
                return {"error": "'Close' saknas i nedladdad data."}
            
            df = df["Close"]
        else:
            return {"error": "Förväntade MultiIndex med 'Close', men fick annat format."}

        df = df.dropna()
        returns = df.pct_change().dropna()

        # Förbered regression
        y = returns[req.symbol]
        X = returns[req.benchmark]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        # Konvertera till listor för frontend-graf
        scatter_data = pd.DataFrame({
            "etf_ret": y.values,
            "benchmark_ret": X[req.benchmark].values
        }).dropna()

        return {
            "symbol": req.symbol,
            "benchmark": req.benchmark,
            "alpha": round(model.params["const"], 5),
            "beta": round(model.params[req.benchmark], 4),
            "r_squared": round(model.rsquared, 4),
            "p_alpha": round(model.pvalues["const"], 4),
            "p_beta": round(model.pvalues[req.benchmark], 4),
            "n_obs": int(model.nobs),
            "scatter": scatter_data.to_dict(orient="list")
        }


    except Exception as e:
        return {"error": str(e)}

@app.post("/etf-factors")
def analyze_fama_french(data: ETFFactorRequest):
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import pandas_datareader.data as web

    try:
        # 1. Ladda ETF-priser och beräkna avkastningar
        etf_data = yf.download(data.symbol, start=data.start, end=data.end, progress=False)
        if etf_data.empty:
            return {"error": f"No data found for symbol {data.symbol}"}
            
        etf_returns = etf_data['Close'].pct_change()

        # 2. Hämta Fama-French-faktorer (dagliga)
        try:
            ff = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench")[0]
        except Exception as e:
            return {"error": f"Failed to fetch Fama-French factors: {str(e)}"}
            
        # Convert index to datetime
        ff.index = pd.to_datetime(ff.index)
        etf_returns.index = pd.to_datetime(etf_returns.index)
        
        # 3. Align the data using pandas
        aligned_data = pd.concat([etf_returns, ff], axis=1, join='inner')
        if aligned_data.empty:
            return {"error": "No overlapping dates found between ETF and factor data"}
            
        # Remove any remaining NaN values
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < 30:  # Minimum sample size check
            return {"error": "Insufficient data points for analysis"}
            
        # 4. Prepare data for regression
        y = aligned_data.iloc[:, 0] - aligned_data['RF'] / 100  # Excess returns
        X = aligned_data[['Mkt-RF', 'SMB', 'HML']].values / 100  # Convert factors to decimals
        X = sm.add_constant(X)  # Add constant for alpha

        # 5. Run regression
        model = sm.OLS(y, X).fit()

        # 6. Extract and convert results to Python scalars
        params = model.params.tolist()
        pvalues = model.pvalues.tolist()

        return {
            "symbol": data.symbol,
            "alpha": round(params[0], 5),
            "beta_market": round(params[1], 4),
            "beta_smb": round(params[2], 4),
            "beta_hml": round(params[3], 4),
            "r_squared": round(float(model.rsquared), 4),
            "p_alpha": round(pvalues[0], 4),
            "p_market": round(pvalues[1], 4),
            "p_smb": round(pvalues[2], 4),
            "p_hml": round(pvalues[3], 4),
            "n_obs": int(model.nobs)
        }

    except Exception as e:
        return {"error": str(e)}

