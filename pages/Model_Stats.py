import webbrowser
import streamlit as st

YT_VIDEO = ''
GITHUB_LINK = ''

st.title("Custom Trained Models Stats")

tab1, tab2 = st.tabs(["MODEL1: Freshness Model", "MODEL2: Quality Inspector"])

with st.sidebar:
    st.info(f"""QUALITY CONTROL ASSISTANT 📝
            Team CIFAR version v4.2""")
    st.sidebar.image("image.jpg", use_column_width=True)

    if st.button('Working Video', use_container_width=True):
        webbrowser.open_new_tab(YT_VIDEO)
    
    if st.button('Github Repository', use_container_width=True):
        webbrowser.open_new_tab(GITHUB_LINK)

#-------------------------------------------MODEL 1----------------------------------------

with tab1:
    st.header("Fruit and Vegetable Freshness Model")

    col1, col2 = st.columns([1, 5])

    with col1:
        st.subheader("Dataset:")
    with col2:
        st.success("10K images comprised of various fruits and vegetables, containing both fresh and rotten produce")

    st.subheader("Training Model: COCOn Object Detection")
    st.image('pages/model.png', channels="BGR", use_column_width=True)

    st.subheader("Training Graphs:")

    col1, col2 = st.columns(2)
    
    with col1:
        st.image("pages/map.png", channels="BGR", use_column_width=True)

    with col2:
        st.image("pages/stats.png", channels="BGR", use_column_width=True)

    st.image("pages/graph.png", channels="BGR", use_column_width=True)

#-------------------------------------------MODEL 2----------------------------------------

with tab2:
    st.header("Bottle/Can Quality Inspector (Bonus Task)")

    col1, col2 = st.columns([1, 5])

    with col1:
        st.subheader("Dataset:")
    with col2:
        st.success("1500 self captured images comprised of various pictures of bottles and cans with several defects")

    st.subheader("Training Model: COCOn-seg Segmentation Model")
    st.image('pages/model_2.png', channels="BGR", use_column_width=True)

    st.subheader("Training Graphs:")

    col1, col2 = st.columns(2)
    
    with col1:
        st.image("pages/map_2.png", channels="BGR", use_column_width=True)

    with col2:
        st.image("pages/graph_2.png", channels="BGR", use_column_width=True)

    st.image("pages/stats_2.png", channels="BGR", use_column_width=True)
