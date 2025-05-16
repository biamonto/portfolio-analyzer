import streamlit as st
import requests
import plotly.graph_objects as go
import datetime

st.set_page_config(
    page_title="Fund analysis",
    page_icon="📊"
)

st.title("📊 Fund analysis")

with st.form("etf_form"):
    symbol = st.text_input("Fund symbol (ex: SPY, QQQ, ARKK)").upper()
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Startdatum", value=datetime.date(2020, 1, 1))
    with col2:
        end = st.date_input("Slutdatum", value=datetime.date.today())
    submitted = st.form_submit_button("Analysera ETF")

if submitted and symbol:
    payload = {
        "symbol": symbol,
        "start": str(start),
        "end": str(end)
    }
    res = requests.post("http://localhost:8000/etf-analysis", json=payload)

    if res.ok:
        data = res.json()
        if "error" in data:
            st.error(data["error"])
        else:
            st.subheader(f"🔍 Resultat för {data['symbol']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Totalavkastning", f"{round(data['total_return'] * 100, 2)} %")
            col2.metric("CAGR", f"{round(data['cagr'] * 100, 2)} %")
            col3.metric("% positiva månader", f"{round(data['positive_months'] * 100, 1)} %")

            col4, col5, col6 = st.columns(3)
            col4.metric("Sharpe", data['sharpe'])
            col5.metric("Sortino", data['sortino'])
            col6.metric("Standardavvikelse", f"{round(data['std_dev'] * 100, 2)} %")

            col7, _, _ = st.columns(3)
            col7.metric("Max drawdown", f"{round(data['max_drawdown'] * 100, 2)} %")

            # Prisgraf
            st.subheader("📈 Prisgraf")
            if "history" in data:
                dates = list(data["history"].keys())
                prices = list(data["history"].values())
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name="Pris"))
                fig.update_layout(title=f"{data['symbol']} – Pris", xaxis_title="Datum", yaxis_title="USD")
                st.plotly_chart(fig)

                st.subheader("🧠 Alpha- och beta-analys (CAPM)")

                # Default benchmark är SPY – du kan göra detta redigerbart sen
                payload_alpha = {
                    "symbol": symbol,
                    "benchmark": "SPY",
                    "start": str(start),
                    "end": str(end)
                }
                alpha_res = requests.post("http://localhost:8000/etf-alpha", json=payload_alpha)

                if alpha_res.ok:
                    alpha_data = alpha_res.json()
                    if "error" in alpha_data:
                        st.error(alpha_data["error"])
                    else:
                        st.markdown(f"**Benchmark:** {alpha_data['benchmark']}")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Alpha", f"{round(alpha_data['alpha'] * 100, 3)} %")
                        col2.metric("Beta", round(alpha_data['beta'], 2))
                        col3.metric("R²", round(alpha_data['r_squared'], 2))

                        st.caption(f"📊 {alpha_data['n_obs']} dagliga observationer")

                        if alpha_data["p_alpha"] < 0.05:
                            st.success(f"✅ Alpha är statistiskt signifikant (p={alpha_data['p_alpha']})")
                        else:
                            st.info(f"ℹ️ Alpha ej signifikant (p={alpha_data['p_alpha']})")

                        if alpha_data["beta"] > 1.3:
                            st.warning("⚠️ Hög marknadsexponering (beta > 1.3)")

                        st.subheader("📉 Avkastning: ETF vs Benchmark")

                        if "scatter" in alpha_data:
                            import plotly.express as px
                            scatter = alpha_data["scatter"]
                            fig_scatter = px.scatter(
                                x=scatter["benchmark_ret"],
                                y=scatter["etf_ret"],
                                labels={"x": f"{symbol} benchmark (daglig %)", "y": f"{symbol} (daglig %)"},
                                trendline="ols",
                                title="Daglig ETF-avkastning vs Benchmark"
                            )
                            st.plotly_chart(fig_scatter)

                else:
                    st.error("Kunde inte hämta alpha-analys.")

                st.subheader("🧪 Faktoranalys (Fama-French 3-faktor)")

                payload_ff = {
                    "symbol": symbol,
                    "start": str(start),
                    "end": str(end)
                }

                ff_res = requests.post("http://localhost:8000/etf-factors", json=payload_ff)

                if ff_res.ok:
                    ff_data = ff_res.json()
                    if "error" in ff_data:
                        st.error(ff_data["error"])
                    else:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Alpha", f"{round(ff_data['alpha']*100, 3)} %")
                        col2.metric("Beta (Marknad)", round(ff_data["beta_market"], 3))
                        col3.metric("R²", round(ff_data["r_squared"], 3))

                        col4, col5 = st.columns(2)
                        col4.metric("Beta (SMB)", round(ff_data["beta_smb"], 3))
                        col5.metric("Beta (HML)", round(ff_data["beta_hml"], 3))

                        st.caption(f"📊 {ff_data['n_obs']} dagliga observationer")

                        if ff_data["p_alpha"] < 0.05:
                            st.success(f"✅ Alpha signifikant (p={ff_data['p_alpha']})")
                        else:
                            st.info(f"ℹ️ Alpha ej signifikant (p={ff_data['p_alpha']})")

                        if abs(ff_data["beta_smb"]) > 0.3:
                            st.info(f"📈 Small cap-exponering: {round(ff_data['beta_smb'], 2)}")

                        if abs(ff_data["beta_hml"]) > 0.3:
                            st.info(f"📘 Value- eller growth-bias: {round(ff_data['beta_hml'], 2)}")
                else:
                    st.error("Kunde inte hämta faktordata.")

    else:
        st.error("Något gick fel med API-anropet.")
