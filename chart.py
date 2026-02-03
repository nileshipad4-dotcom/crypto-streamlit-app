import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.title("📈 Bitcoin Price Chart")

# Date selector
start_date = st.date_input("Start date", value=None)

# Download BTC data
btc = yf.download("BTC-USD", start=start_date)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(btc.index, btc["Close"])
ax.set_title("Bitcoin Price (USD)")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.grid(True)

# Show in Streamlit
st.pyplot(fig)
