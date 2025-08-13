# Portfolio Analysis App

An interactive toolkit to understand risk, performance, and portfolio characteristics. Compare ETFs, evaluate factor exposures, build portfolios, and test optimization strategies.

## Features

- **ETF Analysis**: Quantitatively evaluate ETFs and funds using alpha, beta, factor exposures and risk metrics
- **Portfolio Optimization**: Construct and optimize your own portfolio based on historical returns and risk models
- **Risk Analysis**: Calculate Value at Risk (VaR) and other risk metrics
- **Factor Analysis**: Fama-French 3-factor model analysis
- **Portfolio Characteristics**: Geographic, sector, and currency exposure analysis

## Local Development

### Prerequisites
- Python 3.8 or higher
- pip

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd portfolio-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run frontend/app.py
```

4. Open your browser and go to `http://localhost:8501`

## Deployment to Streamlit Cloud

### Prerequisites
- GitHub repository with your code
- Streamlit Cloud account

### Steps

1. **Push your code to GitHub**:
   - Make sure your repository is public or you have a paid Streamlit Cloud account
   - Ensure all files are committed and pushed

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to: `frontend/app.py`
   - Click "Deploy"

### Important Notes for Deployment

- ✅ **Pure Streamlit**: The app is now a pure Streamlit application (no separate backend needed)
- ✅ **Caching**: Data fetching is cached for better performance
- ✅ **Dependencies**: All required packages are in `requirements.txt`
- ✅ **Configuration**: Streamlit settings are in `.streamlit/config.toml`

## Project Structure

```
portfolio-app/
├── frontend/
│   ├── app.py                 # Main Streamlit app
│   └── pages/
│       ├── fund_analysis.py   # ETF analysis page
│       └── portfolio_analysis.py # Portfolio optimization page
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── README.md                 # This file
```

## Dependencies

- **streamlit**: Web app framework
- **yfinance**: Financial data fetching
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **statsmodels**: Statistical analysis
- **pandas_datareader**: Fama-French factor data
- **plotly**: Interactive charts
- **pypfopt**: Portfolio optimization
- **httpx**: HTTP client for API calls

## Usage

### ETF Analysis
1. Enter a fund symbol (e.g., SPY, QQQ, ARKK)
2. Select start and end dates
3. Click "Run" to see:
   - Performance metrics (returns, Sharpe ratio, etc.)
   - Price chart
   - Alpha/Beta analysis vs SPY
   - Fama-French 3-factor analysis

### Portfolio Analysis
1. Enter asset symbols separated by commas
2. Set asset weights (percentages)
3. Choose analysis options:
   - **Analyze**: Get expected return, volatility, and Sharpe ratio
   - **Optimize**: Find optimal weights using maximum Sharpe ratio
   - **Historical Returns**: View portfolio performance over time
   - **Portfolio Characteristics**: See geographic, sector, and currency exposure
   - **Risk Analysis**: Calculate VaR and CVaR

## Author

Built by **Markus Biamont**

## License

This project is open source and available under the MIT License.
