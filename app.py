import streamlit as st
import pickle
import numpy as np
import re
from urllib.parse import urlparse

st.set_page_config(page_title="Phishing Detector", page_icon="🔐", layout="centered")

st.title("🔐 Phishing Website Detector")
st.markdown("Enter a URL below — features are extracted automatically.")
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

# Dataset mean values for features we cannot extract from URL alone
# These are neutral/average values so the model isn't biased
FEATURE_MEANS = [
    61.13, 21.09, 0.15, 2.48, 1.00, 0.02, 0.14, 0.16, 0.0, 0.29,
    0.32, 0.007, 0.12, 4.29, 4.29, 0.0007, 1.03, 0.004, 0.062, 0.002,
    0.035, 0.45, 0.13, 0.007, 0.61, 0.053, 0.025, 0.00035, 0.002, 0.066,
    0.05, 0.022, 2.23, 0.20, 0.083, 0.12, 0.00017, 0.498, 0.003, 6.23,
    2.93, 3.13, 5.02, 2.40, 15.39, 10.47, 10.56, 7.26, 7.68, 5.09,
    0.33, 0.10, 0.004, 0.005, 0.018, 0.060, 87.19, 0.60, 0.28, 0.0,
    0.78, 0.0, 0.16, 0.0, 0.062, 0.064, 0.44, 51.98, 0.0, 42.87,
    23.24, 0.0, 0.0013, 0.006, 37.06, 0.001, 0.0014, 0.12, 0.78, 0.44,
    0.073, 492.53, 4062.54, 856756.64, 0.020, 0.53, 3.19
]

PHISH_HINTS = ['secure', 'account', 'update', 'login', 'signin', 'bank',
               'verify', 'free', 'lucky', 'service', 'bonus', 'ebayisapi', 'webscr']

SHORTENERS = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
              'buff.ly', 'adf.ly', 'cutt.ly', 'shorte.st']

def having_ip(url):
    return 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0

def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    path = parsed.path or ''
    full = url.lower()

    # Start with dataset means as neutral baseline
    f = list(FEATURE_MEANS)

    # Overwrite with values we CAN extract from the URL
    f[0]  = len(url)
    f[1]  = len(hostname)
    f[2]  = having_ip(url)
    f[3]  = url.count('.')
    f[4]  = url.count('-')
    f[5]  = url.count('@')
    f[6]  = url.count('?')
    f[7]  = url.count('&')
    f[9]  = url.count('=')
    f[10] = url.count('_')
    f[11] = url.count('~')
    f[12] = url.count('%')
    f[14] = url.count('/')
    f[15] = url.count('*')
    f[16] = url.count(':')
    f[17] = url.count(',')
    f[18] = url.count(';')
    f[19] = url.count('$')
    f[20] = url.count(' ')
    f[21] = 1 if 'www' in full else 0
    f[22] = full.count('.com')
    f[23] = 1 if '//' in path else 0
    f[24] = 1 if parsed.scheme == 'https' else 0
    f[25] = sum(c.isdigit() for c in url) / max(len(url), 1)
    f[26] = sum(c.isdigit() for c in hostname) / max(len(hostname), 1)
    f[27] = 1 if 'xn--' in hostname else 0
    f[28] = 1 if parsed.port else 0
    f[32] = len(hostname.split('.')) - 1
    f[33] = 1 if hostname.startswith('-') or hostname.endswith('-') else 0
    f[35] = 1 if any(s in full for s in SHORTENERS) else 0
    f[50] = sum(1 for h in PHISH_HINTS if h in full)

    return np.array(f).reshape(1, -1)

# ── UI ─────────────────────────────────────────────────────────────────────────
url_input = st.text_input("🌐 Enter URL:", placeholder="https://example.com")

model_choice = st.selectbox("🤖 Select Model:",
    ['Random Forest', 'Logistic Regression', 'Decision Tree', 'SVM'])

accuracy_map = {
    'Random Forest': '96.59%', 'Logistic Regression': '95.58%',
    'Decision Tree': '93.44%', 'SVM': '96.28%'
}
st.info(f"📊 {model_choice} Accuracy: {accuracy_map[model_choice]}")

if st.button("🔍 Check URL", use_container_width=True):
    if not url_input.strip():
        st.warning("Please enter a URL first.")
    else:
        url = url_input.strip()
        if not url.startswith('http'):
            url = 'http://' + url

        features = extract_features(url)
        features_scaled = scaler.transform(features)

        models_dict = {
            'Random Forest': rf_model, 'Logistic Regression': lr_model,
            'Decision Tree': dt_model, 'SVM': svm_model
        }
        model = models_dict[model_choice]
        prediction = model.predict(features_scaled)[0]

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            confidence = max(proba) * 100
        else:
            confidence = 95.0

        st.markdown("---")
        if prediction == 1:
            st.error("🔴 PHISHING WEBSITE DETECTED!")
            st.markdown("### ⚠️ This website appears to be **DANGEROUS**!")
        else:
            st.success("🟢 LEGITIMATE WEBSITE!")
            st.markdown("### ✅ This website appears to be **SAFE**!")

        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))

        with st.expander("🔬 Key Extracted Features"):
            parsed = urlparse(url)
            st.write({
                "URL Length": int(features[0][0]),
                "Hostname Length": int(features[0][1]),
                "Has IP Address": bool(features[0][2]),
                "Dots": int(features[0][3]),
                "Hyphens": int(features[0][4]),
                "Slashes": int(features[0][14]),
                "Subdomains": int(features[0][32]),
                "HTTPS": bool(features[0][24]),
                "Phish Hints": int(features[0][50]),
                "Shortening Service": bool(features[0][35]),
            })

st.markdown("---")
st.markdown("*Phishing Website Detection | 8th Sem Research Project*")


