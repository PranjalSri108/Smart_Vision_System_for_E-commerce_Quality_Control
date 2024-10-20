import streamlit as st

# Title for the app
st.title("Embedded Website in Streamlit")

# Embedding an external website using an iframe
url = "https://demo.roboflow.com/grid_6.0_dataset-qrd2r/2?publishable_key=rf_CzzWnlDIYYfT9AlYaYLXCH0D22l2"
st.components.v1.html(f'<iframe src="{url}" width="800" height="400"></iframe>', height=600)
