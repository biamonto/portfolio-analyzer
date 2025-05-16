import streamlit as st
import requests
import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

        # Prepare API data
        portfolio = {
            "assets": [{"symbol": k, "weight": round(v, 4)} for k, v in weights.items()],
            "start": str(start),
            "end": str(end)
        }

    if st.button("🔍 Analyze portfolio"):
        res = requests.post("http://localhost:8000/analyze", json=portfolio)
        if res.ok:
            result = res.json()
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Analysis complete!")
                st.write("📈 Expected return:", round(result["expected_return"] * 100, 2), "%")
                st.write("📉 Volatility:", round(result["volatility"] * 100, 2), "%")
                st.write("📊 Sharpe ratio:", result["sharpe_ratio"])
        else:
            st.error("Could not analyze portfolio.")

    if st.button("🧠 Optimize portfolio"):
        res = requests.post("http://localhost:8000/optimize", json=portfolio)
        if res.ok:
            result = res.json()
            if "error" in result:
                st.error(result["error"])
            else:
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

with st.expander("📈 Historical returns"):
    if st.button("Show history"):
        res = requests.post("http://localhost:8000/history", json=portfolio)
        if res.ok:
            data = res.json()
            if "error" in data:
                st.error(data["error"])
            else:
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
            st.error("Could not fetch history.")

with st.expander("🧬 Show portfolio characteristics"):
    if st.button("Analyse"):
        res = requests.post("http://localhost:8000/portfolio-characteristics", json=portfolio)
        if res.ok:
            data = res.json()

            st.subheader("💱 FX exposure")
            if data["currency_weights"]:
                fig_currency = px.pie(
                    names=list(data["currency_weights"].keys()),
                    values=[v * 100 for v in data["currency_weights"].values()],
                    title="FX (%)"
                )
                st.plotly_chart(fig_currency)
                # Efter st.plotly_chart(fig_currency)
                main_currency = max(data["currency_weights"], key=data["currency_weights"].get)
                share = data["currency_weights"][main_currency]
                if share > 0.5:
                    st.info(f"🔁 High concentration in {main_currency} – currency hedging may be warranted.")
                    if main_currency == "USD":
                        st.write("💡 Example: FXE (Euro hedge), USD/SEK-forwards")
                    if main_currency == "EUR":
                        st.write("💡 Example: EUO, EUR/SEK-forwards")

            else:
                st.warning("Ingen valutadata hittades.")

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

with st.expander("📉 Risk Analysis: Value at Risk (VaR)"):
    horizon = st.slider("Choose time horizon (days)", 1, 20, 1)
    if st.button("Calculate VaR"):
        request_body = portfolio.copy()
        request_body["horizon"] = horizon
        res = requests.post("http://localhost:8000/var", json=request_body)
        if res.ok:
            data = res.json()
            if "error" in data:
                st.error(data["error"])
            else:
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
            st.error("Could not fetch VaR data.")
