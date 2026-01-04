import streamlit as st

st.set_page_config(
    page_title="Test App",
    layout="wide"
)

st.title("✅ Streamlit is working")

if st.button("Click me"):
    st.write("Button works")

