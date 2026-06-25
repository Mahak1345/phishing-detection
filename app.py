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

# ── Feature extractor ──────────────────────────────────────────────────────────
PHISH_HINTS = ['secure', 'account', 'update', 'login', 'signin', 'bank',
                'verify', 'free', 'lucky', 'service', 'bonus', 'ebayisapi', 'webscr']

SHORTENERS = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
              'buff.ly', 'adf.ly', 'cutt.ly', 'shorte.st']

def having_ip_address(url):
    pattern = r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])'
    return 1 if re.search(pattern, url) else 0

def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    path = parsed.path or ''
    full = url.lower()

    features = np.zeros(87)

    features[0]  = len(url)                                        # length_url
    features[1]  = len(hostname)                                   # length_hostname
    features[2]  = having_ip_address(url)                          # ip
    features[3]  = url.count('.')                                  # nb_dots
    features[4]  = url.count('-')                                  # nb_hyphens
    features[5]  = url.count('@')                                  # nb_at
    features[6]  = url.count('?')                                  # nb_qm
    features[7]  = url.count('&')                                  # nb_and
    features[8]  = url.count('|')                                  # nb_or
    features[9]  = url.count('=')                                  # nb_eq
    features[10] = url.count('_')                                  # nb_underscore
    features[11] = url.count('~')                                  # nb_tilde
    features[12] = url.count('%')                                  # nb_percent
    features[13] = url.count('/')                                  # nb_slash (use index 14 below too)
    features[14] = url.count('/')                                  # nb_slash
    features[15] = url.count('*')                                  # nb_star
    features[16] = url.count(':')                                  # nb_colon
    features[17] = url.count(',')                                  # nb_comma
    features[18] = url.count(';')                                  # nb_semicolumn
    features[19] = url.count('$')                                  # nb_dollar
    features[20] = url.count(' ')                                  # nb_space
    features[21] = 1 if 'www' in full else 0                       # nb_www
    features[22] = full.count('.com')                              # nb_com
    features[23] = 1 if '//' in path else 0                       # nb_dslash
    features[24] = 1 if 'https' in parsed.scheme else 0            # https_token (scheme is https)
    features[25] = sum(c.isdigit() for c in url) / max(len(url),1) # ratio_digits_url
    features[26] = sum(c.isdigit() for c in hostname) / max(len(hostname),1)  # ratio_digits_host
    features[27] = 1 if 'xn--' in hostname else 0                 # punycode
    features[28] = 1 if parsed.port else 0                        # port
    features[29] = 1 if re.search(r'\.(php|html|htm|asp|aspx|jsp)$', path) else 0  # path_extension
    features[32] = len(hostname.split('.')) - 1                    # nb_subdomains (approx)
    features[33] = 1 if hostname.startswith('-') or hostname.endswith('-') else 0  # prefix_suffix
    features[36] = 1 if any(s in full for s in SHORTENERS) else 0  # shortening_service
    features[51] = sum(1 for h in PHISH_HINTS if h in full)        # phish_hints

    return features.reshape(1, -1)

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

        with st.expander("🔬 Extracted Features (key ones)"):
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
                "Phish Hints": int(features[0][51]),
                "Shortening Service": bool(features[0][36]),
            })

st.markdown("---")
st.markdown("*Phishing Website Detection | 8th Sem Research Project*")

