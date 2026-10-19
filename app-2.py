import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

st.set_page_config(
    page_title="Indian Portfolio Risk Engine",
    page_icon="📈",
    layout="wide"
)

st.title("Indian Portfolio Risk & Allocation Engine")
st.caption(
    "Explore historical returns, portfolio risk, and "
    "alternative allocation strategies using NSE-listed stocks."
)

TICKERS = [
    "BHARTIARTL.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "LT.NS",
    "RELIANCE.NS",
    "SBIN.NS",
    "TCS.NS"
]

st.sidebar.header("Portfolio settings")

selected = st.sidebar.multiselect(
    "Select stocks",
    TICKERS,
    default=TICKERS
)

period = st.sidebar.selectbox(
    "Historical period",
    ["1y", "2y", "3y", "5y"],
    index=3
)

if len(selected) < 2:
    st.warning("Select at least two stocks to compare portfolios.")
    st.stop()

@st.cache_data(ttl=3600)
def download_prices(tickers, period):
    data = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return pd.DataFrame()
        data = data["Close"]
    else:
        if "Close" not in data.columns:
            return pd.DataFrame()
        data = data[["Close"]]
        data.columns = list(tickers)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    return data.dropna(how="all").ffill().dropna()

prices = download_prices(tuple(selected), period)

if prices.empty or len(prices) < 30:
    st.error(
        "Not enough price data was downloaded. "
        "Try again later or select different stocks."
    )
    st.stop()

# Daily percentage returns
returns = prices.pct_change().dropna()

# Remove any stocks with unusable return data
returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

if returns.empty or len(returns) < 30:
    st.error("Not enough valid return observations for analysis.")
    st.stop()

# Portfolio weights
n = len(returns.columns)

equal_weights = pd.Series(
    1 / n,
    index=returns.columns
)

volatility = returns.std()
inverse_weights = 1 / volatility.replace(0, np.nan)
inverse_weights = inverse_weights.dropna()

if inverse_weights.empty:
    st.error("Could not calculate valid inverse-volatility weights.")
    st.stop()

inverse_weights = inverse_weights / inverse_weights.sum()

# Keep the equal-weight comparison on the same usable stocks
returns = returns[inverse_weights.index]
equal_weights = pd.Series(
    1 / len(returns.columns),
    index=returns.columns
)

# Calculate portfolio daily returns
portfolio_returns = pd.DataFrame({
    "Equal Weight": returns.dot(equal_weights),
    "Inverse Volatility": returns.dot(inverse_weights)
})

portfolio_growth = (1 + portfolio_returns).cumprod() * 100

# Portfolio metrics
rows = []

for strategy in portfolio_returns.columns:
    r = portfolio_returns[strategy]
    wealth = portfolio_growth[strategy]
    drawdown = wealth / wealth.cummax() - 1

    annual_return = r.mean() * 252
    annual_vol = r.std() * np.sqrt(252)
    sharpe = (
        annual_return / annual_vol
        if annual_vol > 0 else np.nan
    )

    rows.append({
        "Strategy": strategy,
        "Annualised Return (%)": annual_return * 100,
        "Annualised Volatility (%)": annual_vol * 100,
        "Maximum Drawdown (%)": drawdown.min() * 100,
        "Historical 95% Daily VaR (%)":
            -r.quantile(0.05) * 100,
        "Sharpe Ratio (0% risk-free rate)": sharpe
    })

summary = pd.DataFrame(rows).set_index("Strategy")

# Headline metrics
st.subheader("Portfolio risk and performance")

col1, col2, col3, col4 = st.columns(4)

equal = summary.loc["Equal Weight"]
inverse = summary.loc["Inverse Volatility"]

col1.metric(
    "Equal-weight return",
    f"{equal['Annualised Return (%)']:.2f}%"
)
col2.metric(
    "Inverse-volatility return",
    f"{inverse['Annualised Return (%)']:.2f}%"
)
col3.metric(
    "Equal-weight volatility",
    f"{equal['Annualised Volatility (%)']:.2f}%"
)
col4.metric(
    "Inverse-volatility volatility",
    f"{inverse['Annualised Volatility (%)']:.2f}%"
)

st.dataframe(
    summary.round(2),
    use_container_width=True
)

st.caption(
    "Annualised returns are arithmetic estimates based on daily returns. "
    "Historical results do not predict future performance."
)

# Portfolio growth chart
st.subheader("Historical portfolio growth")

fig = px.line(
    portfolio_growth,
    title="Growth of ₹100 under Each Strategy",
    labels={
        "value": "Portfolio value (₹)",
        "variable": "Strategy",
        "Date": "Date"
    }
)
fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Allocation comparison
st.subheader("Portfolio allocations")

allocation = pd.DataFrame({
    "Equal Weight (%)": equal_weights * 100,
    "Inverse Volatility (%)": inverse_weights * 100
}).round(2)

st.dataframe(allocation, use_container_width=True)

# Correlation heatmap
st.subheader("Stock return correlations")

corr = returns.corr()

corr_fig = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu_r",
    title="Correlation of Daily Stock Returns"
)

st.plotly_chart(corr_fig, use_container_width=True)

# Download results
st.subheader("Download results")

st.download_button(
    "Download portfolio metrics (CSV)",
    data=summary.round(4).to_csv().encode("utf-8"),
    file_name="portfolio_risk_summary.csv",
    mime="text/csv"
)

st.download_button(
    "Download allocation weights (CSV)",
    data=allocation.to_csv().encode("utf-8"),
    file_name="portfolio_allocations.csv",
    mime="text/csv"
)

st.info(
    "This tool is for educational analysis, not personalised investment "
    "advice. It excludes transaction costs, taxes, and live trading."
)
