import streamlit as st
from views import show_open_loop, show_mpc, show_tests

st.set_page_config(page_title="Diafiltration MPC", layout="wide")

# Sidebar
st.sidebar.image("assets/tank_image.png", caption="Diafiltration Tank", use_container_width=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Open-loop", "MPC", "Test"])

# Routing
if page == "Open-loop":
    show_open_loop()
elif page == "MPC":
    show_mpc()
elif page == "Test":
    show_tests()

