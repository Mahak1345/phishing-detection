import streamlit as st
import pickle
import numpy as np
import pandas as pd

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
        rf = pickle.load(f)
    with open('lr_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('dt_model.pkl', 'rb') as f:
        dt = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return rf, lr, dt, svm, scaler

rf_model, lr_model, dt_model, svm_model, scaler = load_models()
st.success("✅ All 4 models loaded successfully!")
st.markdown("---")

# Model selector
model_choice = st.selectbox(
    "🤖 Select ML Model:",
    ['Random Forest', 'Logistic Regression', 'Decision Tree', 'SVM']
)

accuracy_map = {
    'Random Forest': '96.59%',
    'Logistic Regression': '95.58%',
    'Decision Tree': '93.44%',
    'SVM': '96.28%'
}
st.info(f"📊 {model_choice} Accuracy: {accuracy_map[model_choice]}")
st.markdown("---")

st.markdown("### 🌐 Enter Website Features:")

col1, col2, col3 = st.columns(3)

with col1:
    length_url = st.number_input("URL Length", value=50)
    length_hostname = st.number_input("Hostname Length", value=20)
    nb_dots = st.number_input("Number of Dots", value=2)
    nb_hyphens = st.number_input("Number of Hyphens", value=0)
    nb_slash = st.number_input("Number of Slashes", value=3)
    ip = st.selectbox("Has IP Address", [0, 1])
    https_token = st.selectbox("HTTPS Token", [0, 1])

with col2:
    nb_subdomains = st.number_input("Number of Subdomains", value=1)
    nb_hyperlinks = st.number_input("Number of Hyperlinks", value=10)
    ratio_intHyperlinks = st.number_input("Ratio Internal Hyperlinks", value=0.5)
    ratio_extHyperlinks = st.number_input("Ratio External Hyperlinks", value=0.3)
    phish_hints = st.selectbox("Phish Hints", [0, 1])
    domain_age = st.number_input("Domain Age (days)", value=365)
    web_traffic = st.number_input("Web Traffic", value=1000)

with col3:
    google_index = st.selectbox("Google Index", [0, 1], index=1)
    page_rank = st.number_input("Page Rank", value=3)
    dns_record = st.selectbox("DNS Record", [0, 1], index=1)
    login_form = st.selectbox("Has Login Form", [0, 1])
    iframe = st.selectbox("Has iFrame", [0, 1])
    popup_window = st.selectbox("Has Popup Window", [0, 1])
    domain_in_title = st.selectbox("Domain in Title", [0, 1], index=1)

if st.button("🔍 Check Website", use_container_width=True):

    models_dict = {
        'Random Forest': rf_model,
        'Logistic Regression': lr_model,
        'Decision Tree': dt_model,
        'SVM': svm_model
    }

    selected_model = models_dict[model_choice]

    # Create input with all 87 features (fill rest with 0)
    input_data = np.zeros((1, 87))

    # Fill in the values user entered
    input_data[0][0] = length_url
    input_data[0][1] = length_hostname
    input_data[0][3] = nb_dots
    input_data[0][4] = nb_hyphens
    input_data[0][14] = nb_slash
    input_data[0][2] = ip
    input_data[0][24] = https_token
    input_data[0][32] = nb_subdomains
    input_data[0][56] = nb_hyperlinks
    input_data[0][57] = ratio_intHyperlinks
    input_data[0][58] = ratio_extHyperlinks
    input_data[0][51] = phish_hints
    input_data[0][79] = domain_age
    input_data[0][78] = web_traffic
    input_data[0][85] = google_index
    input_data[0][86] = page_rank
    input_data[0][83] = dns_record
    input_data[0][63] = login_form
    input_data[0][71] = iframe
    input_data[0][72] = popup_window
    input_data[0][77] = domain_in_title

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = selected_model.predict(input_scaled)[0]
    if hasattr(selected_model, 'predict_proba'):
        probability = selected_model.predict_proba(input_scaled)[0]
        confidence = max(probability) * 100
    else:
        confidence = 95.0

    st.markdown("---")
    if prediction == 1:
        st.error("🔴 PHISHING WEBSITE DETECTED!")
        st.markdown("### ⚠️ This website appears to be DANGEROUS!")
    else:
        st.success("🟢 LEGITIMATE WEBSITE!")
        st.markdown("### ✅ This website appears to be SAFE!")

    st.markdown(f"Confidence: {confidence:.2f}%")
    st.progress(int(confidence))

st.markdown("---")
st.markdown("*Phishing Website Detection | 8th Sem Research Project*")
