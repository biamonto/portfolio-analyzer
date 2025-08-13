import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web

st.set_page_config(
    page_title="Fund analysis",
    page_icon="📊"
)

# Cache data fetching functions for better performance
@st.cache_data
def fetch_stock_data(symbol, start, end):
    """Fetch stock data with caching"""
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"].dropna()
        else:
            df = df["Close"].dropna()
        return pd.DataFrame(df).rename(columns={symbol: "Price", "Close": "Price"})
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

@st.cache_data
def fetch_benchmark_data(symbol, benchmark, start, end):
    """Fetch both ETF and benchmark data"""
    try:
        df = yf.download([symbol, benchmark], start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" not in df.columns.levels[0]:
                return None
            df = df["Close"]
        else:
            return None
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching benchmark data: {str(e)}")
        return None

@st.cache_data
def fetch_fama_french_factors(start, end):
    """Fetch Fama-French factors"""
    try:
        ff = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench")[0]
        ff.index = pd.to_datetime(ff.index)
        return ff[(ff.index >= start) & (ff.index <= end)]
    except Exception as e:
        st.error(f"Error fetching Fama-French factors: {str(e)}")
        return None

def analyze_etf(df, symbol):
    """Analyze ETF performance metrics"""
    if df is None or df.empty:
        return None
    
    returns = df["Price"].pct_change().dropna()
    
    # Basic calculations
    total_return = (df["Price"].iloc[-1] / df["Price"].iloc[0]) - 1
    days = (df.index[-1] - df.index[0]).days
    cagr = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
    std_dev = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Sortino ratio
    downside = returns[returns < 0]
    sortino = returns.mean() / downside.std() * np.sqrt(252) if not downside.empty and downside.std() > 0 else None
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_dd = drawdown.min()
    
    # % positive months
    monthly = df["Price"].resample("M").last().pct_change().dropna()
    pos_months = (monthly > 0).sum() / len(monthly) if len(monthly) > 0 else 0
    
    def safe_round(x, decimals=4):
        if pd.isna(x) or np.isinf(x):
            return None
        return round(x, decimals)
    
    return {
        "symbol": symbol.upper(),
        "total_return": safe_round(total_return),
        "cagr": safe_round(cagr),
        "sharpe": safe_round(sharpe, 2),
        "sortino": safe_round(sortino, 2) if sortino else None,
        "std_dev": safe_round(std_dev),
        "max_drawdown": safe_round(max_dd),
        "positive_months": safe_round(pos_months, 4),
        "history": df["Price"].round(2).to_dict()
    }

def analyze_alpha_beta(df, symbol, benchmark):
    """Analyze alpha and beta using CAPM"""
    if df is None or df.empty:
        return None
    
    returns = df.pct_change().dropna()
    
    # Prepare regression
    y = returns[symbol]
    X = returns[benchmark]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    # Convert to lists for frontend graph
    scatter_data = pd.DataFrame({
        "etf_ret": y.values,
        "benchmark_ret": X[benchmark].values
    }).dropna()
    
    return {
        "symbol": symbol,
        "benchmark": benchmark,
        "alpha": round(model.params["const"], 5),
        "beta": round(model.params[benchmark], 4),
        "r_squared": round(model.rsquared, 4),
        "p_alpha": round(model.pvalues["const"], 4),
        "p_beta": round(model.pvalues[benchmark], 4),
        "n_obs": int(model.nobs),
        "scatter": scatter_data.to_dict(orient="list")
    }

def analyze_fama_french(etf_returns, symbol, start, end):
    """Analyze Fama-French 3-factor model"""
    try:
        # Fetch Fama-French factors
        ff = fetch_fama_french_factors(start, end)
        if ff is None:
            return None
        
        # Align the data
        etf_returns.index = pd.to_datetime(etf_returns.index)
        aligned_data = pd.concat([etf_returns, ff], axis=1, join='inner')
        if aligned_data.empty:
            return {"error": "No overlapping dates found between ETF and factor data"}
        
        # Remove any remaining NaN values
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < 30:
            return {"error": "Insufficient data points for analysis"}
        
        # Prepare data for regression
        y = aligned_data.iloc[:, 0] - aligned_data['RF'] / 100  # Excess returns
        X = aligned_data[['Mkt-RF', 'SMB', 'HML']].values / 100  # Convert factors to decimals
        X = sm.add_constant(X)  # Add constant for alpha
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Extract results
        params = model.params.tolist()
        pvalues = model.pvalues.tolist()
        
        return {
            "symbol": symbol,
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

st.title("📊 Fund analysis")

with st.form("etf_form"):
    symbol = st.text_input("Fund symbol (ex: SPY, QQQ, ARKK)").upper()
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", value=datetime.date(2020, 1, 1))
    with col2:
        end = st.date_input("End date", value=datetime.date.today())
    submitted = st.form_submit_button("Run")

if submitted and symbol:
    # Fetch and analyze ETF data
    df = fetch_stock_data(symbol, start, end)
    data = analyze_etf(df, symbol)
    
    if data:
        st.subheader(f"🔍 Result for {data['symbol']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total return", f"{round(data['total_return'] * 100, 2)} %")
        col2.metric("CAGR", f"{round(data['cagr'] * 100, 2)} %")
        col3.metric("% positive months", f"{round(data['positive_months'] * 100, 1)} %")

        col4, col5, col6 = st.columns(3)
        col4.metric("Sharpe", data['sharpe'])
        col5.metric("Sortino", data['sortino'])
        col6.metric("Standard deviation", f"{round(data['std_dev'] * 100, 2)} %")

        col7, _, _ = st.columns(3)
        col7.metric("Max drawdown", f"{round(data['max_drawdown'] * 100, 2)} %")

        # Price chart
        st.subheader("📈 Price")
        if "history" in data:
            dates = list(data["history"].keys())
            prices = list(data["history"].values())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name="Price"))
            fig.update_layout(title=f"{data['symbol']} – Price", xaxis_title="Date", yaxis_title="USD")
            st.plotly_chart(fig)

            st.subheader("🧠 Alpha and beta analysis (CAPM)")

            # Alpha/Beta analysis with SPY benchmark
            benchmark_df = fetch_benchmark_data(symbol, "SPY", start, end)
            alpha_data = analyze_alpha_beta(benchmark_df, symbol, "SPY")

            if alpha_data:
                st.markdown(f"**Benchmark:** {alpha_data['benchmark']}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Alpha", f"{round(alpha_data['alpha'] * 100, 3)} %")
                col2.metric("Beta", round(alpha_data['beta'], 2))
                col3.metric("R²", round(alpha_data['r_squared'], 2))

                st.caption(f"📊 {alpha_data['n_obs']} daily observations")

                if alpha_data["p_alpha"] < 0.05:
                    st.success(f"✅ Alpha is significant (p={alpha_data['p_alpha']})")
                else:
                    st.info(f"ℹ️ Alpha not significant (p={alpha_data['p_alpha']})")

                if alpha_data["beta"] > 1.3:
                    st.warning("⚠️ High market exposure (beta > 1.3)")

                st.subheader("📉 Return: ETF vs Benchmark")

                if "scatter" in alpha_data:
                    scatter = alpha_data["scatter"]
                    fig_scatter = px.scatter(
                        x=scatter["benchmark_ret"],
                        y=scatter["etf_ret"],
                        labels={"x": f"{symbol} benchmark (daily %)", "y": f"{symbol} (daily %)"},
                        trendline="ols",
                        title="Daily ETF return vs Benchmark"
                    )
                    st.plotly_chart(fig_scatter)
            else:
                st.error("Could not perform alpha analysis.")

            st.subheader("🧪 Factor analysis (Fama-French 3 factor)")

            # Fama-French analysis
            etf_returns = df["Price"].pct_change().dropna()
            ff_data = analyze_fama_french(etf_returns, symbol, start, end)

            if ff_data and "error" not in ff_data:
                col1, col2, col3 = st.columns(3)
                col1.metric("Alpha", f"{round(ff_data['alpha']*100, 3)} %")
                col2.metric("Beta (Market)", round(ff_data["beta_market"], 3))
                col3.metric("R²", round(ff_data["r_squared"], 3))

                col4, col5 = st.columns(2)
                col4.metric("Beta (SMB)", round(ff_data["beta_smb"], 3))
                col5.metric("Beta (HML)", round(ff_data["beta_hml"], 3))

                st.caption(f"📊 {ff_data['n_obs']} daily observations")

                if ff_data["p_alpha"] < 0.05:
                    st.success(f"✅ Alpha significant (p={ff_data['p_alpha']})")
                else:
                    st.info(f"ℹ️ Alpha not significant (p={ff_data['p_alpha']})")

                if abs(ff_data["beta_smb"]) > 0.3:
                    st.info(f"📈 Small cap exposure: {round(ff_data['beta_smb'], 2)}")

                if abs(ff_data["beta_hml"]) > 0.3:
                    st.info(f"📘 Value or growth bias: {round(ff_data['beta_hml'], 2)}")
            elif ff_data and "error" in ff_data:
                st.error(ff_data["error"])
            else:
                st.error("Could not fetch factor data.")
    else:
        st.error("Could not analyze the ETF. Please check the symbol and date range.")
