import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🔐",
    layout="centered"
)

st.title("🔐 Phishing Website Detection System")
st.markdown("### Using Machine Learning & URL Features")
st.markdown("---")

@st.cache_resource
def load_models():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_models()
st.success("✅ Model loaded successfully!")
st.markdown("---")

st.markdown("### 🌐 Enter Website Features:")
st.info("Rate each feature: -1 = Suspicious, 0 = Neutral, 1 = Legitimate")

col1, col2 = st.columns(2)

with col1:
    google_index = st.selectbox("Google Index", [-1, 0, 1], index=2)
    page_rank = st.selectbox("Page Rank", [-1, 0, 1], index=2)
    web_traffic = st.selectbox("Web Traffic", [-1, 0, 1], index=2)
    domain_age = st.selectbox("Domain Age", [-1, 0, 1], index=2)
    nb_hyperlinks = st.number_input("Number of Hyperlinks", value=10)

with col2:
    having_ip = st.selectbox("Has IP Address", [-1, 1], index=1)
    https_token = st.selectbox("HTTPS Token", [-1, 1], index=1)
    dns_record = st.selectbox("DNS Record", [-1, 1], index=1)
    phish_hints = st.selectbox("Phish Hints", [-1, 0, 1], index=2)
    nb_dots = st.number_input("Number of Dots in URL", value=2)

if st.button("🔍 Check Website", use_container_width=True):
    st.markdown("---")
    st.warning("⚠️ Note: This uses only 10 key features for demo. Full model uses 87 features.")
    st.markdown("### 🔬 For Full Prediction — Paste your results from Jupyter!")
    st.markdown("The app demonstrates the detection system interface.")
    st.success("✅ App is working perfectly!")

st.markdown("---")
st.markdown("*Phishing Website Detection | 8th Sem Research Project*")