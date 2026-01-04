import streamlit as st

def reboot_app():
    """
    HARD reset:
    - clears all Streamlit caches
    - clears session_state
    - forces full script rerun
    """
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.rerun()
