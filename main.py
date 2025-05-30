import streamlit as st
import joblib
import pickle
import numpy as np

# Configure page
st.set_page_config(
    page_title="MedPredict AI", 
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Compact CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 0; max-width: none; background: #fafbfc; }
    .stDeployButton { display: none; }
    header[data-testid="stHeader"] { display: none; }
    
    .premium-navbar {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
        padding: 1rem 2rem;
        position: sticky;
        top: 0;
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .nav-brand h1 {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        text-align: center;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 0 0 1rem;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        margin: 0 0 2rem;
        opacity: 0.9;
    }
    
    .disease-header {
        background: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .hepatitis-header { background: linear-gradient(135deg, #fef3c7 0%, #f59e0b 100%); color: #78350f; }
    .hiv-header { background: linear-gradient(135deg, #ddd6fe 0%, #7c3aed 100%); color: #581c87; }
    .tb-header { background: linear-gradient(135deg, #d1fae5 0%, #059669 100%); color: #064e3b; }
    
    .form-section {
        background: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f2937;
        margin: 0 0 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
        width: 100%;
        height: 3rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }
    
    .result-card {
        background: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .hero-subtitle { font-size: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    models = {}
    try:
        models['hepatitis'] = pickle.load(open("hepatitis_model.pkl", "rb"))
    except FileNotFoundError:
        st.error("Hepatitis model not found")
        models['hepatitis'] = None
    
    try:
        models['hiv'] = joblib.load("hiv_model.pkl")
        models['vectorizer'] = joblib.load("vectorizer.pkl")
    except FileNotFoundError:
        st.error("HIV model/vectorizer not found")
        models['hiv'] = models['vectorizer'] = None
    
    try:
        models['tb'] = joblib.load("tb_predictor_model.pkl")
    except FileNotFoundError:
        st.error("TB model not found")
        models['tb'] = None
    
    return models

models = load_models()

# Navigation
st.markdown("""
<div class="premium-navbar">
    <div class="nav-brand">
        <h1>🔬 MedPredict AI</h1>
    </div>
    <div style="color: #10b981; font-weight: 500;">● System Active</div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">Medical AI Prediction Suite</h1>
    <p class="hero-subtitle">Advanced disease risk assessment using machine learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🩺 Prediction Models")
    tab = st.radio(
        "Choose Analysis:",
        ["🫀 Hepatitis", "🧪 HIV Risk", "🫁 TB Screening"]
    )
    
    st.markdown("---")
    st.markdown("*Trained on clinical datasets*\n*Validated by medical experts*")

# Helper functions
def map_bool(val): 
    return {'False': 0, 'True': 1, 'Unknown': -1}[val]

def map_sex(val): 
    return 0 if val == "male" else 1

# HEPATITIS TAB
if tab == "🫀 Hepatitis":
    st.markdown("""
    <div class="disease-header hepatitis-header">
        <h2>🫀 Hepatitis Prognosis Analysis</h2>
        <p>Liver health assessment using clinical parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    if models['hepatitis'] is None:
        st.error("Hepatitis model not available")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="form-section"><h4 class="section-title">Demographics</h4>', unsafe_allow_html=True)
            age = st.slider("Age", 7, 78, 40)
            sex = st.selectbox("Sex", ["male", "female"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="form-section"><h4 class="section-title">Treatment</h4>', unsafe_allow_html=True)
            steroid = st.selectbox("Steroid Therapy", ["False", "True", "Unknown"])
            antivirals = st.selectbox("Antiviral Therapy", ["False", "True"])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="form-section"><h4 class="section-title">Symptoms</h4>', unsafe_allow_html=True)
            fatigue = st.selectbox("Fatigue", ["False", "True", "Unknown"])
            malaise = st.selectbox("Malaise", ["False", "True", "Unknown"])
            anorexia = st.selectbox("Loss of Appetite", ["False", "True", "Unknown"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="form-section"><h4 class="section-title">Examination</h4>', unsafe_allow_html=True)
            liver_big = st.selectbox("Hepatomegaly", ["False", "True", "Unknown"])
            liver_firm = st.selectbox("Liver Firmness", ["False", "True", "Unknown"])
            spleen_palpable = st.selectbox("Splenomegaly", ["False", "True", "Unknown"])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="form-section"><h4 class="section-title">Clinical Signs</h4>', unsafe_allow_html=True)
            spiders = st.selectbox("Spider Angiomata", ["False", "True", "Unknown"])
            ascites = st.selectbox("Ascites", ["False", "True", "Unknown"])
            varices = st.selectbox("Esophageal Varices", ["False", "True", "Unknown"])
            histology = st.selectbox("Histology", ["False", "True"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="form-section"><h4 class="section-title">Lab Values</h4>', unsafe_allow_html=True)
            bilirubin = st.number_input("Bilirubin (mg/dL)", 0.3, 8.0, 1.0, step=0.1)
            alk_phosphate = st.number_input("Alk Phosphatase (U/L)", 26, 295, 85)
            sgot = st.number_input("AST/SGOT (U/L)", 14, 648, 25)
            albumin = st.number_input("Albumin (g/dL)", 2.1, 6.4, 4.0, step=0.1)
            protime = st.number_input("Prothrombin Time (%)", 0, 100, 85)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔍 Generate Prognosis", type="primary"):
            with st.spinner("Processing..."):
                features = [
                    age, map_sex(sex), map_bool(steroid), map_bool(antivirals),
                    map_bool(fatigue), map_bool(malaise), map_bool(anorexia), map_bool(liver_big),
                    map_bool(liver_firm), map_bool(spleen_palpable), map_bool(spiders),
                    map_bool(ascites), map_bool(varices), map_bool(histology),
                    bilirubin, alk_phosphate, sgot, albumin, protime
                ]
                pred = models['hepatitis'].predict([features])[0]
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                if pred == 1:
                    st.success("✅ **FAVORABLE PROGNOSIS** - Positive survival indicators")
                    st.info("📋 Continue current therapy, routine monitoring")
                else:
                    st.error("⚠️ **CONCERNING PROGNOSIS** - Intensive intervention required")
                    st.warning("🚨 Escalate to specialist, intensive monitoring")
                st.markdown('</div>', unsafe_allow_html=True)

# HIV TAB
elif tab == "🧪 HIV Risk":
    st.markdown("""
    <div class="disease-header hiv-header">
        <h2>🧪 HIV Risk Assessment</h2>
        <p>Symptom analysis for HIV-related manifestations</p>
    </div>
    """, unsafe_allow_html=True)

    if models['hiv'] is None or models['vectorizer'] is None:
        st.error("HIV model not available")
    else:
        symptoms_list = [
            "Persistent Fever", "Night Sweats", "Chronic Fatigue", "Weight Loss", 
            "Persistent Diarrhea", "Lymphadenopathy", "Skin Manifestations", "Oral Candidiasis",
            "Cognitive Impairment", "Neurological Symptoms", "Opportunistic Infections", "Low CD4 Count",
            "Elevated Viral Load", "Recurrent Infections"
        ]

        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-title">📋 Symptom Assessment</h4>', unsafe_allow_html=True)
        
        with st.form("hiv_form"):
            cols = st.columns(3)
            selected = []
            
            for idx, symptom in enumerate(symptoms_list):
                with cols[idx % 3]:
                    if st.checkbox(symptom, key=f"hiv_{idx}"):
                        selected.append(symptom)
            
            submitted = st.form_submit_button("🔍 Assess Risk", type="primary")

            if submitted:
                if not selected:
                    st.warning("⚠️ Please select symptoms to proceed")
                else:
                    with st.spinner("Analyzing..."):
                        vec = models['vectorizer'].transform([", ".join(selected)])
                        pred = models['hiv'].predict(vec)[0]
                        prob = models['hiv'].predict_proba(vec)[0]
                        
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        
                        if pred == 1:
                            st.error(f"🚨 **HIGH RISK** - Confidence: {max(prob):.1%}")
                            st.error("Order HIV testing, refer to specialist")
                        else:
                            st.success(f"✅ **LOW RISK** - Confidence: {max(prob):.1%}")
                            st.info("Continue routine monitoring")
                        
                        with st.expander("Selected Symptoms"):
                            for i, symptom in enumerate(selected, 1):
                                st.write(f"{i}. {symptom}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# TB TAB
elif tab == "🫁 TB Screening":
    st.markdown("""
    <div class="disease-header tb-header">
        <h2>🫁 Tuberculosis Screening</h2>
        <p>WHO-validated TB screening criteria</p>
    </div>
    """, unsafe_allow_html=True)

    if models['tb'] is None:
        st.error("TB model not available")
    else:
        tb_symptoms = [
            "Prolonged Fever (≥2 weeks)", "Hemoptysis", "Blood-tinged Sputum", "Night Sweats",
            "Chest Pain", "Back Pain", "Dyspnea", "Weight Loss",
            "Fatigue", "Cervical Lymphadenopathy",
            "Productive Cough (2-4 weeks)", "Generalized Lymphadenopathy", "Anorexia"
        ]

        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-title">🩺 TB Screening</h4>', unsafe_allow_html=True)
        
        tb_inputs = []
        cols = st.columns(2)
        
        for idx, symptom in enumerate(tb_symptoms):
            with cols[idx % 2]:
                response = st.selectbox(
                    f"{symptom}",
                    ["Absent", "Present"], 
                    key=f"tb_{idx}"
                )
                tb_inputs.append(1 if response == "Present" else 0)

        if st.button("🔍 Execute TB Analysis", type="primary"):
            with st.spinner("Processing..."):
                tb_pred = models['tb'].predict([tb_inputs])[0]
                tb_proba = models['tb'].predict_proba([tb_inputs])[0][tb_pred]
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                if tb_pred == 1:
                    st.error(f"🚨 **HIGH TB RISK** - Probability: {tb_proba:.2%}")
                    st.error("Seek immediate clinical evaluation")
                else:
                    st.success(f"✅ **LOW TB RISK** - Probability: {tb_proba:.2%}")
                    st.info("TB unlikely based on screening")

                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**⚠️ Disclaimer:** This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.")
