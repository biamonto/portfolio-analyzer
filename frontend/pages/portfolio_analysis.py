import streamlit as st
import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
from collections import defaultdict
import httpx

# Cache data fetching functions for better performance
@st.cache_data
def fetch_portfolio_data(symbols, start, end):
    """Fetch portfolio data with caching"""
    try:
        raw = yf.download(symbols, start=start, end=end, progress=False)
        if raw.empty:
            return None
        
        # Handle MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                df = raw['Close']
            except KeyError:
                return None
        else:
            # Single asset
            try:
                df = raw[['Close']]
                df.columns = [symbols[0]]
            except KeyError:
                return None
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching portfolio data: {str(e)}")
        return None

def analyze_portfolio(df, weights_dict):
    """Analyze portfolio performance"""
    if df is None or df.empty:
        return None
    
    try:
        # Calculate returns
        returns = df.pct_change().dropna()
        
        # Calculate expected returns (mean)
        mu = returns.mean() * 252  # Annualized
        
        # Calculate covariance matrix
        S = returns.cov() * 252  # Annualized
        
        # Calculate portfolio metrics
        portfolio_return = sum(mu[asset] * weights_dict[asset] for asset in weights_dict.keys())
        
        # Calculate portfolio variance
        portfolio_variance = 0
        for asset1 in weights_dict.keys():
            for asset2 in weights_dict.keys():
                portfolio_variance += weights_dict[asset1] * weights_dict[asset2] * S.loc[asset1, asset2]
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            "expected_return": round(portfolio_return, 4),
            "volatility": round(portfolio_volatility, 4),
            "sharpe_ratio": round(sharpe_ratio, 2),
        }
    except Exception as e:
        st.error(f"Error analyzing portfolio: {str(e)}")
        return None

def optimize_portfolio(df):
    """Optimize portfolio using maximum Sharpe ratio"""
    if df is None or df.empty:
        return None
    
    try:
        # Calculate returns
        returns = df.pct_change().dropna()
        
        # Calculate expected returns (mean)
        mu = returns.mean() * 252  # Annualized
        
        # Calculate covariance matrix
        S = returns.cov() * 252  # Annualized
        
        # Number of assets
        n_assets = len(df.columns)
        
        # Risk-free rate
        risk_free_rate = 0.02
        
        # Define objective function (negative Sharpe ratio to minimize)
        def objective(weights):
            portfolio_return = np.sum(mu * weights)
            portfolio_variance = np.dot(weights.T, np.dot(S, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            return -sharpe_ratio
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            # Create weights dictionary
            weights_dict = {}
            for i, asset in enumerate(df.columns):
                weights_dict[asset] = round(result.x[i], 4)
            
            # Clean weights (remove very small weights)
            cleaned_weights = {}
            for asset, weight in weights_dict.items():
                if weight > 0.001:  # Only keep weights > 0.1%
                    cleaned_weights[asset] = weight
            
            # Renormalize if needed
            total_weight = sum(cleaned_weights.values())
            if total_weight > 0:
                for asset in cleaned_weights:
                    cleaned_weights[asset] = round(cleaned_weights[asset] / total_weight, 4)
            
            return cleaned_weights
        else:
            st.error("Optimization failed")
            return None
            
    except Exception as e:
        st.error(f"Error optimizing portfolio: {str(e)}")
        return None

def get_portfolio_history(df, weights_dict):
    """Get portfolio historical performance"""
    if df is None or df.empty:
        return None
    
    try:
        # Normalize each asset price to 1.0 at the beginning
        df_normalized = df / df.iloc[0]
        
        # Weight according to user's input
        for col in df_normalized.columns:
            df_normalized[col] *= weights_dict.get(col, 0)
        
        # Sum up to the portfolio's daily value
        df_normalized["Total"] = df_normalized.sum(axis=1)
        
        return {
            "dates": df_normalized.index.strftime("%Y-%m-%d").tolist(),
            "values": df_normalized["Total"].round(4).tolist()
        }
    except Exception as e:
        st.error(f"Error calculating portfolio history: {str(e)}")
        return None

@st.cache_data
def get_portfolio_characteristics(symbols, weights):
    """Get portfolio characteristics using Finnhub API"""
    FINNHUB_API_KEY = "d0h5cu9r01qv1u35cv5gd0h5cu9r01qv1u35cv60"
    
    currency_weights = defaultdict(float)
    sector_weights = defaultdict(float)
    region_weights = defaultdict(float)
    
    try:
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
    except Exception as e:
        st.error(f"Error fetching portfolio characteristics: {str(e)}")
        return None

def calculate_var(df, weights_dict, horizon=1):
    """Calculate Value at Risk"""
    if df is None or df.empty:
        return None
    
    try:
        returns = df.pct_change().dropna()
        
        for col in returns.columns:
            returns[col] *= weights_dict.get(col, 0)
        
        portfolio_returns = returns.sum(axis=1)
        
        # Aggregate over desired horizon
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
    except Exception as e:
        st.error(f"Error calculating VaR: {str(e)}")
        return None

st.title("📊 Portfolio analysis and optimization")

symbols = st.text_input("Enter assets (e.g. AAPL, GOOGL, MSFT)").upper()
start = st.date_input("Start date", datetime.date(2023, 1, 1))
end = st.date_input("End date", datetime.date.today())

weights = {}
total_input = 0
if symbols:
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    st.subheader("Set asset weights manually (%):")

    for s in symbol_list:
        weight_input = st.number_input(
            f"{s}", min_value=0.0, max_value=100.0,
            value=round(100 / len(symbol_list), 2), step=0.1,
            key=f"{s}_weight"
        )
        weights[s] = weight_input / 100
        total_input += weight_input

    # Show summary and feedback
    st.markdown(f"**Total weight: {round(total_input, 2)} %**")
    if total_input < 99.9:
        st.warning("⚠️ Sum < 100% – weights will be normalized automatically.")
    elif total_input > 100.1:
        st.warning("⚠️ Sum > 100% – weights will be scaled down.")
    else:
        st.success("✅ Sum of weights equal 100%.")

    # Normalize to 100%
    total_fraction = sum(weights.values())
    for k in weights:
        weights[k] /= total_fraction

    # Fetch portfolio data
    df = fetch_portfolio_data(symbol_list, start, end)

    if st.button("🔍 Analyze portfolio"):
        if df is not None:
            result = analyze_portfolio(df, weights)
            if result:
                st.success("Analysis complete!")
                st.write("📈 Expected return:", round(result["expected_return"] * 100, 2), "%")
                st.write("📉 Volatility:", round(result["volatility"] * 100, 2), "%")
                st.write("📊 Sharpe ratio:", result["sharpe_ratio"])
            else:
                st.error("Could not analyze portfolio.")
        else:
            st.error("No data found for selected symbols and dates.")

    if st.button("🧠 Optimize portfolio"):
        if df is not None:
            result = optimize_portfolio(df)
            if result:
                st.success("Optimization complete!")
                st.subheader("Optimal weights:")

                labels = list(result.keys())
                values = [v * 100 for v in result.values()]
                fig = px.pie(names=labels, values=values, title="Optimized portfolio allocation")
                st.plotly_chart(fig)

                for k, v in result.items():
                    st.write(f"{k}: {round(v * 100, 2)} %")
            else:
                st.error("Could not optimize portfolio.")
        else:
            st.error("No data found for selected symbols and dates.")

with st.expander("📈 Historical returns"):
    if st.button("Show history"):
        if df is not None:
            data = get_portfolio_history(df, weights)
            if data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data["dates"],
                    y=data["values"],
                    mode="lines",
                    name="Portfolio value",
                    line=dict(color="royalblue")
                ))
                fig.update_layout(
                    title="Portfolio return",
                    xaxis_title="Date",
                    yaxis_title="Relative value (start = 1.0)",
                    height=400
                )
                st.plotly_chart(fig)
            else:
                st.error("Could not calculate portfolio history.")
        else:
            st.error("No data found for selected symbols and dates.")

with st.expander("🧬 Show portfolio characteristics"):
    if st.button("Analyse"):
        if df is not None:
            data = get_portfolio_characteristics(symbol_list, weights)
            if data:
                st.subheader("💱 FX exposure")
                if data["currency_weights"]:
                    fig_currency = px.pie(
                        names=list(data["currency_weights"].keys()),
                        values=[v * 100 for v in data["currency_weights"].values()],
                        title="FX (%)"
                    )
                    st.plotly_chart(fig_currency)
                    
                    main_currency = max(data["currency_weights"], key=data["currency_weights"].get)
                    share = data["currency_weights"][main_currency]
                    if share > 0.5:
                        st.info(f"🔁 High concentration in {main_currency} – currency hedging may be warranted.")
                        if main_currency == "USD":
                            st.write("💡 Example: FXE (Euro hedge), USD/SEK-forwards")
                        if main_currency == "EUR":
                            st.write("💡 Example: EUO, EUR/SEK-forwards")
                else:
                    st.warning("No currency data found.")

                st.subheader("🏦 Sector allocation")
                if data["sector_weights"]:
                    fig_sector = px.pie(
                        names=list(data["sector_weights"].keys()),
                        values=[v * 100 for v in data["sector_weights"].values()],
                        title="Sector allocation (%)"
                    )
                    st.plotly_chart(fig_sector)
                    top_sector = max(data["sector_weights"], key=data["sector_weights"].get)
                    top_sector_pct = data["sector_weights"][top_sector]
                    if top_sector_pct > 0.5:
                        st.info(f"📌 Portfolio is heavily exposed to sector: {top_sector}")
                    else:
                        st.success("✅ Portfolio has balanced sector allocation.")
                else:
                    st.warning("No sector data found.")

                st.subheader("🌍 Regional allocation")
                if data["region_weights"]:
                    fig_region = px.bar(
                        x=list(data["region_weights"].keys()),
                        y=[v * 100 for v in data["region_weights"].values()],
                        labels={"x": "Region", "y": "Share (%)"},
                        title="Geographic distribution"
                    )
                    st.plotly_chart(fig_region)
                    top_region = max(data["region_weights"], key=data["region_weights"].get)
                    region_share = data["region_weights"][top_region]
                    if region_share > 0.6:
                        st.warning(f"🌍 High geographic concentration: {top_region} ({round(region_share*100)} %)")
                    else:
                        st.success("🌎 Portfolio is geographically diversified.")
                else:
                    st.warning("No regional data found.")
            else:
                st.error("Could not fetch portfolio characteristics.")
        else:
            st.error("No data found for selected symbols and dates.")

with st.expander("📉 Risk Analysis: Value at Risk (VaR)"):
    horizon = st.slider("Choose time horizon (days)", 1, 20, 1)
    if st.button("Calculate VaR"):
        if df is not None:
            data = calculate_var(df, weights, horizon)
            if data:
                st.metric(f"📉 {horizon}-day VaR (95%)", f"{round(data['VaR_95'] * 100, 2)} %")
                st.metric(f"🔥 {horizon}-day CVaR", f"{round(data['CVaR_95'] * 100, 2)} %")
                st.metric("📈 Average return", f"{round(data['mean_return'] * 100, 2)} %")
                st.metric("📊 Volatility", f"{round(data['std_dev'] * 100, 2)} %")

                if data["VaR_95"] < -0.03:
                    st.warning("⚠️ Portfolio has high downside risk.")
                elif data["VaR_95"] > -0.01:
                    st.success("✅ Portfolio has low historical downside risk.")

                st.caption(f"Analyzed periods: {data['n_obs']}")

                # Histogram
                st.subheader("📊 Return Distribution")
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data["returns"],
                    nbinsx=50,
                    name="Return",
                    marker_color="lightblue",
                    opacity=0.75
                ))
                fig.add_vline(
                    x=data["var_threshold"],
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="VaR 95%",
                    annotation_position="top left"
                )
                fig.update_layout(
                    title=f"{horizon}-day portfolio return",
                    xaxis_title="Return",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig)
            else:
                st.error("Could not calculate VaR.")
        else:
            st.error("No data found for selected symbols and dates.")
