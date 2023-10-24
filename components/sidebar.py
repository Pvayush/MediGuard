import streamlit as st

def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Upload an IMG, JPEG, or PNG file📄\n"
            "2. Check the info tab for accurate data \n"
            "3. Submit to detect liklihood of fraud💬\n"
        )

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "📖MediGuard allows you to upload healthcare bills and claims"
            " to identify potential error or fraud."
        )
        st.markdown(
            "Powered by unsupervised machine learning"
            " trained with healthcare billing data."
            " MediGuard autonomously identifies suspicious patterns"
            " and anomalies in medical bills and reports cost accuracy."
        )
        st.markdown("Made by \n" 
                    "Ayush Puthiyavettle \n"
                    "Satvik Mahendra \n"
                    "Donne Su \n"
                    "Yash Kukrecha \n")
        st.markdown("---")

        #faq()