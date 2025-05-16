import streamlit as st
import requests
import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.title("📊 Portföljanalys och optimering")

symbols = st.text_input("Ange tillgångar (t.ex. AAPL, GOOGL, MSFT)").upper()
start = st.date_input("Startdatum", datetime.date(2023, 1, 1))
end = st.date_input("Slutdatum", datetime.date.today())

weights = {}
total_input = 0
if symbols:
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    st.subheader("Vikta tillgångarna manuellt (%):")

    for s in symbol_list:
        weight_input = st.number_input(
            f"{s}", min_value=0.0, max_value=100.0,
            value=round(100 / len(symbol_list), 2), step=0.1,
            key=f"{s}_weight"
        )
        weights[s] = weight_input / 100
        total_input += weight_input

    # Visa summering och feedback
    st.markdown(f"**Totalvikt: {round(total_input, 2)} %**")
    if total_input < 99.9:
        st.warning("⚠️ Summan är under 100 % – vikterna kommer normaliseras automatiskt.")
    elif total_input > 100.1:
        st.warning("⚠️ Summan överstiger 100 % – vikterna kommer justeras ned.")
    else:
        st.success("✅ Vikterna summerar till 100 %.")

    # Normalisera till 100 %
    total_fraction = sum(weights.values())
    for k in weights:
        weights[k] /= total_fraction

        # Förbered API-data
        portfolio = {
            "assets": [{"symbol": k, "weight": round(v, 4)} for k, v in weights.items()],
            "start": str(start),
            "end": str(end)
        }


    if st.button("🔍 Analysera portfölj"):
        res = requests.post("http://localhost:8000/analyze", json=portfolio)
        if res.ok:
            result = res.json()
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Analys klar!")
                st.write("📈 Förväntad avkastning:", round(result["expected_return"] * 100, 2), "%")
                st.write("📉 Volatilitet:", round(result["volatility"] * 100, 2), "%")
                st.write("📊 Sharpe-kvot:", result["sharpe_ratio"])
        else:
            st.error("Kunde inte analysera portföljen.")

    if st.button("🧠 Optimera portfölj"):
        res = requests.post("http://localhost:8000/optimize", json=portfolio)
        if res.ok:
            result = res.json()
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Optimering klar!")
                st.subheader("Optimerade vikter:")

                st.subheader("Optimerade vikter:")

                labels = list(result.keys())
                values = [v * 100 for v in result.values()]
                fig = px.pie(names=labels, values=values, title="Optimerad portföljfördelning")
                st.plotly_chart(fig)

                for k, v in result.items():
                    st.write(f"{k}: {round(v * 100, 2)} %")
        else:
            st.error("Kunde inte optimera portföljen.")

with st.expander("📈 Visa portföljens historiska utveckling"):
    if st.button("Visa historik"):
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
                    name="Portföljvärde",
                    line=dict(color="royalblue")
                ))
                fig.update_layout(
                    title="Portföljens utveckling",
                    xaxis_title="Datum",
                    yaxis_title="Relativt värde (start = 1.0)",
                    height=400
                )
                st.plotly_chart(fig)
        else:
            st.error("Kunde inte hämta historik.")

with st.expander("🧬 Visa portföljegenskaper"):
    if st.button("Analysera egenskaper"):
        res = requests.post("http://localhost:8000/portfolio-characteristics", json=portfolio)
        if res.ok:
            data = res.json()

            st.subheader("💱 Valutaexponering")
            if data["currency_weights"]:
                fig_currency = px.pie(
                    names=list(data["currency_weights"].keys()),
                    values=[v * 100 for v in data["currency_weights"].values()],
                    title="Valutafördelning (%)"
                )
                st.plotly_chart(fig_currency)
                # Efter st.plotly_chart(fig_currency)
                main_currency = max(data["currency_weights"], key=data["currency_weights"].get)
                share = data["currency_weights"][main_currency]
                if share > 0.5:
                    st.info(f"🔁 Portföljen är kraftigt viktad mot {main_currency}. Överväg hedge mot växelkursrisk.")
                    if main_currency == "USD":
                        st.write("💡 Exempel: FXE (Euro hedge), USD/SEK-terminer")
                    if main_currency == "EUR":
                        st.write("💡 Exempel: EUO, EUR/SEK-terminer")

            else:
                st.warning("Ingen valutadata hittades.")

            st.subheader("🏦 Sektorallokering")
            if data["sector_weights"]:
                fig_sector = px.pie(
                    names=list(data["sector_weights"].keys()),
                    values=[v * 100 for v in data["sector_weights"].values()],
                    title="Sektorfördelning (%)"
                )
                st.plotly_chart(fig_sector)
                top_sector = max(data["sector_weights"], key=data["sector_weights"].get)
                top_sector_pct = data["sector_weights"][top_sector]
                if top_sector_pct > 0.5:
                    st.info(f"📌 Portföljen är tungt exponerad mot sektorn: {top_sector}")
                else:
                    st.success("✅ Portföljen har en balanserad sektorallokering.")
            else:
                st.warning("Ingen sektordata hittades.")

            st.subheader("🌍 Regionfördelning")
            if data["region_weights"]:
                fig_region = px.bar(
                    x=list(data["region_weights"].keys()),
                    y=[v * 100 for v in data["region_weights"].values()],
                    labels={"x": "Region", "y": "Andel (%)"},
                    title="Geografisk fördelning"
                )
                st.plotly_chart(fig_region)
                # Efter st.plotly_chart(fig_region)
                top_region = max(data["region_weights"], key=data["region_weights"].get)
                region_share = data["region_weights"][top_region]
                if region_share > 0.6:
                    st.warning(f"🌍 Du har hög geografisk koncentration: {top_region} ({round(region_share*100)} %)")
                else:
                    st.success("🌎 Portföljen är geografiskt diversifierad.")
            else:
                st.warning("Ingen regiondata hittades.")
        else:
            st.error("Kunde inte hämta portföljegenskaper.")

with st.expander("📉 Riskanalys: Value at Risk (VaR)"):
    horizon = st.slider("Välj tidshorisont (dagar)", 1, 20, 1)
    if st.button("Beräkna VaR"):
        request_body = portfolio.copy()
        request_body["horizon"] = horizon
        res = requests.post("http://localhost:8000/var", json=request_body)
        if res.ok:
            data = res.json()
            if "error" in data:
                st.error(data["error"])
            else:
                st.metric(f"📉 {horizon}-dagars VaR (95 %)", f"{round(data['VaR_95'] * 100, 2)} %")
                st.metric(f"🔥 {horizon}-dagars CVaR", f"{round(data['CVaR_95'] * 100, 2)} %")
                st.metric("📈 Genomsnittlig avkastning", f"{round(data['mean_return'] * 100, 2)} %")
                st.metric("📊 Volatilitet", f"{round(data['std_dev'] * 100, 2)} %")

                if data["VaR_95"] < -0.03:
                    st.warning("⚠️ Portföljen har hög nedåtrisk.")
                elif data["VaR_95"] > -0.01:
                    st.success("✅ Portföljen har låg historisk nedsiderisk.")

                st.caption(f"Analyserade perioder: {data['n_obs']}")

                # Histogram
                st.subheader("📊 Fördelning av avkastningar")
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data["returns"],
                    nbinsx=50,
                    name="Avkastning",
                    marker_color="lightblue",
                    opacity=0.75
                ))
                fig.add_vline(
                    x=data["var_threshold"],
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="VaR 95 %",
                    annotation_position="top left"
                )
                fig.update_layout(
                    title=f"{horizon}-dagars portföljavkastning",
                    xaxis_title="Avkastning",
                    yaxis_title="Frekvens"
                )
                st.plotly_chart(fig)
        else:
            st.error("Kunde inte hämta VaR-data.")

