import streamlit as st
import pickle
import numpy as np
import re
from urllib.parse import urlparse

st.set_page_config(page_title="Phishing Detector", page_icon="🔐", layout="centered")

st.title("🔐 Phishing Website Detector")
st.markdown("Enter a URL — features are extracted automatically.")
st.markdown("---")

with st.expander("ℹ️ About This Application"):
    st.markdown("""
    **How it works:**
    This application accepts a URL as input and automatically extracts **53 URL-based structural features**
    (such as URL length, number of dots, hyphens, subdomains, HTTPS usage, and suspicious keywords)
    to predict whether the website is legitimate or a phishing attempt.

    **Note on Research vs Deployment:**
    The original research model (documented in the thesis) was trained on **87 features** including
    URL-based, domain-based, and webpage-based attributes, achieving an accuracy of **96.76%**.
    
    This deployed application uses a model retrained on **53 URL-only features** (accuracy: **90.77%**)
    to enable real-time prediction directly from a URL — without requiring page scraping or
    external API calls for domain reputation data.

    > This is standard practice in production ML deployment where real-time usability is prioritized.
    """)

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

PHISH_HINTS = ['secure', 'account', 'update', 'login', 'signin', 'bank',
               'verify', 'free', 'lucky', 'service', 'bonus', 'ebayisapi', 'webscr']

SHORTENERS = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
              'buff.ly', 'adf.ly', 'cutt.ly', 'shorte.st']

SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.club', '.work']

def having_ip(url):
    return 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0

def get_words(text):
    return re.split(r'[\W_]+', text)

def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    path = parsed.path or ''
    full = url.lower()
    scheme = parsed.scheme or ''

    # Word-level features
    all_words = get_words(full)
    host_words = get_words(hostname)
    path_words = get_words(path)
    all_words = [w for w in all_words if w]
    host_words = [w for w in host_words if w]
    path_words = [w for w in path_words if w]

    length_words_raw    = len(all_words)
    shortest_words_raw  = min((len(w) for w in all_words), default=0)
    longest_words_raw   = max((len(w) for w in all_words), default=0)
    avg_words_raw       = np.mean([len(w) for w in all_words]) if all_words else 0

    shortest_word_host  = min((len(w) for w in host_words), default=0)
    longest_word_host   = max((len(w) for w in host_words), default=0)
    avg_word_host       = np.mean([len(w) for w in host_words]) if host_words else 0

    shortest_word_path  = min((len(w) for w in path_words), default=0)
    longest_word_path   = max((len(w) for w in path_words), default=0)
    avg_word_path       = np.mean([len(w) for w in path_words]) if path_words else 0

    char_repeat = max(
        (len(list(g)) for _, g in __import__('itertools').groupby(full)),
        default=0
    )

    # Must match exact order from training:
    # url_features list in notebook
    features = [
        len(url),                                                           # length_url
        len(hostname),                                                      # length_hostname
        having_ip(url),                                                     # ip
        url.count('.'),                                                     # nb_dots
        url.count('-'),                                                     # nb_hyphens
        url.count('@'),                                                     # nb_at
        url.count('?'),                                                     # nb_qm
        url.count('&'),                                                     # nb_and
        url.count('|'),                                                     # nb_or
        url.count('='),                                                     # nb_eq
        url.count('_'),                                                     # nb_underscore
        url.count('~'),                                                     # nb_tilde
        url.count('%'),                                                     # nb_percent
        url.count('/'),                                                     # nb_slash
        url.count('*'),                                                     # nb_star
        url.count(':'),                                                     # nb_colon
        url.count(','),                                                     # nb_comma
        url.count(';'),                                                     # nb_semicolumn
        url.count('$'),                                                     # nb_dollar
        url.count(' '),                                                     # nb_space
        1 if 'www' in full else 0,                                          # nb_www
        full.count('.com'),                                                 # nb_com
        1 if '//' in path else 0,                                           # nb_dslash
        1 if 'http' in path else 0,                                         # http_in_path
        1 if scheme == 'https' else 0,                                      # https_token
        sum(c.isdigit() for c in url) / max(len(url), 1),                  # ratio_digits_url
        sum(c.isdigit() for c in hostname) / max(len(hostname), 1),        # ratio_digits_host
        1 if 'xn--' in hostname else 0,                                    # punycode
        1 if parsed.port else 0,                                            # port
        1 if any(full.endswith(t) or ('/' + t[1:] + '/') in full
                 for t in ['.php','.html','.htm','.asp','.aspx']) else 0,  # tld_in_path
        1 if any(t in hostname for t in ['.com','.org','.net','.gov'])
             and hostname.count('.') > 1 else 0,                           # tld_in_subdomain
        1 if re.search(r'(ww\d|www\d)', hostname) else 0,                  # abnormal_subdomain
        len(hostname.split('.')) - 1,                                       # nb_subdomains
        1 if hostname.startswith('-') or hostname.endswith('-') else 0,    # prefix_suffix
        1 if re.search(r'[a-z0-9]{10,}', hostname.split('.')[0] if hostname else '') and
             not re.search(r'(google|facebook|youtube|amazon|microsoft|apple|wiki)', hostname)
             else 0,                                                        # random_domain
        1 if any(s in full for s in SHORTENERS) else 0,                    # shortening_service
        1 if re.search(r'\.(php|html|htm|asp|aspx|jsp)$', path) else 0,   # path_extension
        full.count('//') - 1 if full.count('//') > 1 else 0,              # nb_redirection
        0,                                                                  # nb_external_redirection
        length_words_raw,                                                   # length_words_raw
        char_repeat,                                                        # char_repeat
        shortest_words_raw,                                                 # shortest_words_raw
        shortest_word_host,                                                 # shortest_word_host
        shortest_word_path,                                                 # shortest_word_path
        longest_words_raw,                                                  # longest_words_raw
        longest_word_host,                                                  # longest_word_host
        longest_word_path,                                                  # longest_word_path
        avg_words_raw,                                                      # avg_words_raw
        avg_word_host,                                                      # avg_word_host
        avg_word_path,                                                      # avg_word_path
        sum(1 for h in PHISH_HINTS if h in full),                          # phish_hints
        1 if any(full.endswith(t) for t in SUSPICIOUS_TLDS) else 0,       # suspecious_tld
        0,                                                                  # statistical_report
    ]

    return np.array(features).reshape(1, -1)

# ── UI ─────────────────────────────────────────────────────────────────────────
url_input = st.text_input("🌐 Enter URL:", placeholder="https://example.com")

model_choice = st.selectbox("🤖 Select Model:",
    ['Random Forest', 'Logistic Regression', 'Decision Tree', 'SVM'])

accuracy_map = {
    'Random Forest': '90.77%',
    'Logistic Regression': '85.52%',
    'Decision Tree': '86.96%',
    'SVM': '89.91%'
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
            'Random Forest': rf_model,
            'Logistic Regression': lr_model,
            'Decision Tree': dt_model,
            'SVM': svm_model
        }
        model = models_dict[model_choice]
        prediction = model.predict(features_scaled)[0]

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            confidence = max(proba) * 100
        else:
            decision = model.decision_function(features_scaled)[0]
            confidence = min(abs(decision) * 20 + 50, 99)

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
                "Slashes": int(features[0][13]),
                "Subdomains": int(features[0][32]),
                "HTTPS": bool(features[0][24]),
                "Phish Hints": int(features[0][51]),
                "Shortening Service": bool(features[0][35]),
                "Suspicious TLD": bool(features[0][52]),
            })

st.markdown("---")
st.markdown("*Phishing Website Detection | 8th Sem Research Project*")


