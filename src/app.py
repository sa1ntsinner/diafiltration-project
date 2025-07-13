# app.py

import streamlit as st
from views import show_open_loop, show_mpc, show_tests

# Set up basic page configuration
st.set_page_config(
    page_title="Diafiltration Control",
    layout="wide"  # Use full-width layout
)

# Define the available pages and corresponding view functions
PAGES = {
    "Open-loop": show_open_loop,   # Page 1: Simulate constant or threshold-based control
    "MPC": show_mpc,               # Page 2: Run and compare MPC strategies
    "Tests": show_tests            # Page 3: Run robustness and performance tests
}

# Sidebar: Add image and navigation menu
st.sidebar.image(
    "assets/tank_image.png", 
    caption="Diafiltration Tank", 
    use_container_width=True
)
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))  # Choose a page to view

# Main title and dispatch selected page
st.title("Diafiltration Control Dashboard")
page = PAGES[selection]
page()  # Call the appropriate page function
