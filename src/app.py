import streamlit as st
from views import show_open_loop, show_mpc, show_tests

st.set_page_config(page_title="Diafiltration Control", layout="wide")

PAGES = {
    "Open-loop": show_open_loop,
    "MPC": show_mpc,
    "Tests": show_tests
}


st.sidebar.image("assets/tank_image.png", caption="Diafiltration Tank", use_container_width=True)
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

st.title("Diafiltration Control Dashboard")
page = PAGES[selection]
page()
