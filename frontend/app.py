import streamlit as st

# 🔧 Page configuration
st.set_page_config(
    page_title="Home – Portfolio App",
    page_icon="📊",
    layout="wide"
)

# 💅 Simple styling using HTML/CSS
st.markdown("""
    <style>
        .centered-title {
            text-align: center;
            color: #2C3E50;
        }
        .description {
            text-align: center;
            font-size: 18px;
            color: #555;
            margin-bottom: 30px;
        }
        .stButton>button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# 📌 Title and short intro
st.markdown("<h1 class='centered-title'>📊 Portfolio Analysis App</h1>", unsafe_allow_html=True)
st.markdown("""
    <p class='description'>
        An interactive toolkit to understand risk, performance, and portfolio characteristics.<br>
        Compare ETFs, evaluate factor exposures, build portfolios, and test optimization strategies.
    </p>
""", unsafe_allow_html=True)

# 🚀 Navigation cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔍 ETF Analysis")
    st.markdown("Quantitatively evaluate ETFs and funds using alpha, beta, factor exposures and risk metrics.")
    st.page_link("pages/fund_analysis.py", label="➡️ Go to ETF Analysis", icon="📈")

with col2:
    st.markdown("### 🧮 Portfolio Optimization")
    st.markdown("Construct and optimize your own portfolio based on historical returns and risk models.")
    st.page_link("pages/portfolio_analysis.py", label="➡️ Go to Portfolio Tool", icon="🧮")

# ℹ️ Optional About section
st.markdown("---")
with st.expander("ℹ️ About this app"):
    st.markdown("""
        This web app is built with Streamlit and Python, aiming to make professional-grade analysis tools 
        accessible to individual investors, students, and professionals.

        It allows users to analyze ETFs and funds using factor models, performance metrics, and return simulations.
        The portfolio section lets you construct and optimize your own portfolio using methods such as 
        Modern Portfolio Theory and other risk-adjusted strategies.

        The backend is powered by FastAPI and uses libraries such as:
        - `yfinance` for financial data
        - `PyPortfolioOpt` for optimization
        - `statsmodels` for regression and factor analysis

        The app was built by **Markus Biamont**.  
        The source code is available on [GitHub](https://github.com/biamonto/portfolio-analyzer).
    """)

