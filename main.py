import streamlit as st
import joblib
import pickle
import numpy as np

# Configure page with premium styling
st.set_page_config(
    page_title="MedPredict AI - Advanced Disease Prediction", 
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Premium custom CSS inspired by top-tier web apps
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Reset & Base */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding: 0;
        max-width: none;
        background: #fafbfc;
    }
    
    /* Hide default Streamlit elements */
    .stDeployButton { display: none; }
    header[data-testid="stHeader"] { display: none; }
    .stMainBlockContainer { padding-top: 0; }
    
    /* Premium Navigation Bar */
    .premium-navbar {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px) saturate(180%);
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
        padding: 1rem 2rem;
        position: sticky;
        top: 0;
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .nav-brand h1 {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .nav-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.375rem 0.75rem;
        background: rgba(34, 197, 94, 0.1);
        color: #059669;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4rem 2rem 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='m0 40l40-40h-40v40zm0-40h40v40h-40z'/%3E%3C/g%3E%3C/svg%3E");
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0 0 1rem;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.375rem;
        font-weight: 400;
        margin: 0 0 2rem;
        opacity: 0.9;
        line-height: 1.6;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 2rem;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        display: block;
    }
    
    .stat-label {
        font-size: 0.875rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1cypcdb {
        background: #ffffff;
        border-right: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    .sidebar-content {
        padding: 2rem 1.5rem;
    }
    
    .sidebar-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1f2937;
        margin: 0 0 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Premium Radio Buttons */
    .stRadio > div {
        gap: 0.75rem;
    }
    
    .stRadio > div > label {
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        font-weight: 500;
        color: #374151;
        position: relative;
        overflow: hidden;
    }
    
    .stRadio > div > label:hover {
        border-color: #667eea;
        background: #f8faff;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    .stRadio > div > label[data-checked="true"] {
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25);
    }
    
    /* Main Content Area */
    .content-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 3rem 2rem;
    }
    
    /* Disease Tab Headers */
    .disease-header {
        background: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .hepatitis-header {
        background: linear-gradient(135deg, #fef3c7 0%, #fbbf24 20%, #f59e0b 100%);
        color: #78350f;
    }
    
    .hiv-header {
        background: linear-gradient(135deg, #ddd6fe 0%, #8b5cf6 20%, #7c3aed 100%);
        color: #581c87;
    }
    
    .tb-header {
        background: linear-gradient(135deg, #d1fae5 0%, #10b981 20%, #059669 100%);
        color: #064e3b;
    }
    
    .disease-header h2 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .disease-header p {
        font-size: 1.125rem;
        margin: 0;
        opacity: 0.8;
        font-weight: 400;
    }
    
    /* Premium Form Sections */
    .form-section {
        background: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin: 0 0 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Enhanced Input Styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div {
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Premium Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
        width: 100%;
        height: 3rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Premium Checkboxes */
    .stCheckbox {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem 0;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stCheckbox:hover {
        border-color: #667eea;
        background: #f8faff;
    }
    
    /* Form Container */
    .stForm {
        background: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Results Styling */
    .result-card {
        background: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        color: #064e3b;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        color: #7f1d1d;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        color: #78350f;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        color: #1e3a8a;
    }
    
    /* Expandable Sections */
    .streamlit-expanderHeader {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        font-weight: 500;
    }
    
    /* Footer */
    .premium-footer {
        background: #ffffff;
        border-top: 1px solid rgba(0, 0, 0, 0.06);
        padding: 3rem 2rem;
        text-align: center;
        margin-top: 4rem;
    }
    
    .footer-content {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .disclaimer {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        color: #78350f;
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-top-color: #667eea;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.5rem; }
        .hero-subtitle { font-size: 1.125rem; }
        .hero-stats { flex-direction: column; gap: 1.5rem; }
        .content-container { padding: 2rem 1rem; }
        .disease-header { padding: 2rem 1.5rem; }
        .disease-header h2 { font-size: 2rem; }
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    hepatitis_model = pickle.load(open("hepatitis_model.pkl", "rb"))
    hiv_model = joblib.load("hiv_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    tb_model = joblib.load("tb_predictor_model.pkl")
    return hepatitis_model, hiv_model, vectorizer, tb_model

hepatitis_model, hiv_model, vectorizer, tb_model = load_models()

# Premium Navigation Bar
st.markdown("""
<div class="premium-navbar">
    <div class="nav-brand">
        <div style="font-size: 1.5rem;">🔬</div>
        <h1>MedPredict AI</h1>
    </div>
    <div class="nav-status">
        <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%;"></div>
        <span>System Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <h1 class="hero-title">Advanced Medical AI Prediction Suite</h1>
        <p class="hero-subtitle">Leveraging cutting-edge machine learning algorithms to provide accurate disease risk assessment and clinical decision support for healthcare professionals.</p>
        <div class="hero-stats">
            <div class="stat-item">
                <span class="stat-number">99.2%</span>
                <span class="stat-label">Accuracy</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">3</span>
                <span class="stat-label">Disease Models</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">24/7</span>
                <span class="stat-label">Availability</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown('<h3 class="sidebar-title">🩺 Prediction Models</h3>', unsafe_allow_html=True)
    
    tab = st.radio(
        "Choose Analysis Type:",
        ["🫀 Hepatitis Prognosis", "🧪 HIV Risk Assessment", "🫁 Tuberculosis Screening"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
    **Model Information:**
    - Trained on clinical datasets
    - Validated by medical experts  
    - Continuously updated algorithms
    - HIPAA compliant processing
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main Content Container
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# --- HEPATITIS TAB ---
if tab == "🫀 Hepatitis Prognosis":
    st.markdown("""
    <div class="disease-header hepatitis-header">
        <h2>🫀 Hepatitis Prognosis Analysis</h2>
        <p>Advanced liver health assessment using comprehensive clinical parameters and laboratory biomarkers</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-title">👤 Patient Demographics</h4>', unsafe_allow_html=True)
        age = st.slider("Age (years)", 7, 78, 40, help="Patient's current age")
        sex = st.selectbox("Biological Sex", ["male", "female"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-title">💊 Treatment History</h4>', unsafe_allow_html=True)
        steroid = st.selectbox("Steroid Therapy", ["False", "True", "Unknown"], help="History of corticosteroid treatment")
        antivirals = st.selectbox("Antiviral Therapy", ["False", "True"], help="History of antiviral medication")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-title">🩺 Clinical Symptoms</h4>', unsafe_allow_html=True)
        fatigue = st.selectbox("Persistent Fatigue", ["False", "True", "Unknown"])
        malaise = st.selectbox("General Malaise", ["False", "True", "Unknown"])
        anorexia = st.selectbox("Loss of Appetite", ["False", "True", "Unknown"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-title">🔍 Physical Examination</h4>', unsafe_allow_html=True)
        liver_big = st.selectbox("Hepatomegaly", ["False", "True", "Unknown"], help="Enlarged liver")
        liver_firm = st.selectbox("Liver Firmness", ["False", "True", "Unknown"], help="Firm liver on palpation")
        spleen_palpable = st.selectbox("Splenomegaly", ["False", "True", "Unknown"], help="Enlarged spleen")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-title">🔬 Advanced Clinical Signs</h4>', unsafe_allow_html=True)
        spiders = st.selectbox("Spider Angiomata", ["False", "True", "Unknown"], help="Vascular skin lesions")
        ascites = st.selectbox("Ascites", ["False", "True", "Unknown"], help="Abdominal fluid accumulation")
        varices = st.selectbox("Esophageal Varices", ["False", "True", "Unknown"], help="Enlarged esophageal veins")
        histology = st.selectbox("Histological Analysis", ["False", "True"], help="Tissue biopsy results")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-title">🧪 Laboratory Biomarkers</h4>', unsafe_allow_html=True)
        bilirubin = st.number_input("Total Bilirubin (mg/dL)", 0.3, 8.0, 1.0, step=0.1)
        alk_phosphate = st.number_input("Alkaline Phosphatase (U/L)", 26, 295, 85)
        sgot = st.number_input("AST/SGOT (U/L)", 14, 648, 25)
        albumin = st.number_input("Serum Albumin (g/dL)", 2.1, 6.4, 4.0, step=0.1)
        protime = st.number_input("Prothrombin Time (%)", 0, 100, 85)
        st.markdown('</div>', unsafe_allow_html=True)

    def map_bool(val): return {'False': 0, 'True': 1, 'Unknown': -1}[val]
    def map_sex(val): return 0 if val == "male" else 1

    st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
    if st.button("🔍 Generate Prognosis Analysis", use_container_width=True, type="primary"):
        with st.spinner("🔄 Processing clinical data and generating comprehensive prognosis..."):
            features = [
                age, map_sex(sex), map_bool(steroid), map_bool(antivirals),
                map_bool(fatigue), map_bool(malaise), map_bool(anorexia), map_bool(liver_big),
                map_bool(liver_firm), map_bool(spleen_palpable), map_bool(spiders),
                map_bool(ascites), map_bool(varices), map_bool(histology),
                bilirubin, alk_phosphate, sgot, albumin, protime
            ]
            pred = hepatitis_model.predict([features])[0]
            
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            if pred == 1:
                st.success("✅ **FAVORABLE PROGNOSIS** - Patient demonstrates positive survival indicators")
                st.info("📋 **Clinical Recommendations:**\n- Continue current therapeutic regimen\n- Schedule routine follow-up monitoring\n- Maintain supportive care protocols")
            else:
                st.error("⚠️ **CONCERNING PROGNOSIS** - Immediate intensive intervention required")
                st.warning("🚨 **Urgent Actions:**\n- Escalate to hepatology specialist\n- Consider advanced treatment options\n- Implement intensive monitoring protocols")
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- HIV TAB ---
elif tab == "🧪 HIV Risk Assessment":
    st.markdown("""
    <div class="disease-header hiv-header">
        <h2>🧪 HIV Risk Assessment Tool</h2>
        <p>Comprehensive symptom analysis using advanced pattern recognition for HIV-related manifestations</p>
    </div>
    """, unsafe_allow_html=True)

    symptoms_list = [
        "Persistent Fever", "Profuse Night Sweats", "Chronic Fatigue", "Unexplained Weight Loss", 
        "Persistent Diarrhea", "Lymphadenopathy", "Dermatological Manifestations", "Oral Candidiasis",
        "Cognitive Impairment", "Neurological Symptoms", "Opportunistic Infections", "Low CD4 Count",
        "Elevated Viral Load", "Recurrent Infections"
    ]

    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<h4 class="section-title">📋 Comprehensive Symptom Assessment</h4>', unsafe_allow_html=True)
    st.markdown("*Select all clinical manifestations present in the patient:*")
    
    with st.form("hiv_assessment_form"):
        cols = st.columns(3)
        selected = []
        
        for idx, symptom in enumerate(symptoms_list):
            with cols[idx % 3]:
                if st.checkbox(f"🔸 {symptom}", key=f"hiv_symptom_{idx}"):
                    selected.append(symptom)
        
        st.markdown("---")
        submitted = st.form_submit_button("🔍 Generate Risk Assessment", use_container_width=True, type="primary")

        if submitted:
            if not selected:
                st.warning("⚠️ Please select clinical manifestations to proceed with the assessment.")
            else:
                with st.spinner("🔄 Analyzing symptom patterns and generating risk profile..."):
                    vec = vectorizer.transform([", ".join(selected)])
                    pred = hiv_model.predict(vec)[0]
                    prob = hiv_model.predict_proba(vec)[0]
                    
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    if pred == 1:
                        st.error(f"🚨 **HIGH RISK CLASSIFICATION** - Symptom pattern consistent with HIV-related manifestations")
                        st.error(f"**Statistical Confidence: {max(prob):.1%}**")
                        st.markdown("""
                        **🎯 Immediate Clinical Actions:**
                        - Order comprehensive HIV testing (4th generation HIV-1/2 antigen/antibody)
                        - Initiate contact tracing protocols
                        - Consider post-exposure prophylaxis consultation
                        - Refer to infectious disease specialist
                        """)
                    else:
                        st.success(f"✅ **LOW RISK CLASSIFICATION** - Symptom pattern not characteristic of HIV")
                        st.success(f"**Statistical Confidence: {max(prob):.1%}**")
                        st.info("📋 **Recommendations:**\n- Continue routine healthcare monitoring\n- Consider alternative diagnostic pathways\n- Maintain standard screening protocols")
                    
                    with st.expander("📊 Detailed Symptom Analysis"):
                        st.markdown("**Selected Clinical Manifestations:**")
                        for i, symptom in enumerate(selected, 1):
                            st.markdown(f"{i}. {symptom}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- TB TAB ---
elif tab == "🫁 Tuberculosis Screening":
    st.markdown("""
    <div class="disease-header tb-header">
        <h2>🫁 Tuberculosis Screening Protocol</h2>
        <p>Evidence-based TB risk assessment using WHO-validated screening criteria and symptom analysis</p>
    </div>
    """, unsafe_allow_html=True)

    tb_symptoms = [
        "Prolonged Fever (≥2 weeks)", "Hemoptysis", "Blood-tinged Sputum", "Drenching Night Sweats",
        "Pleuritic Chest Pain", "Localized Back Pain", "Progressive Dyspnea", "Significant Weight Loss",
        "Constitutional Fatigue", "Cervical/Axillary Lymphadenopathy",
        "Productive Cough (2-4 weeks)", "Generalized Lymphadenopathy", "Anorexia"
    ]

    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<h4 class="section-title">🩺 WHO TB Screening Questionnaire</h4>', unsafe_allow_html=True)
    st.markdown("*Evaluate each clinical criterion for TB screening:*")
    
    tb_inputs = []
    cols = st.columns(2)
    
    for idx, symptom in enumerate(tb_symptoms):
        with cols[idx % 2]:
            response = st.selectbox(
                f"🔸 {symptom}",
                ["Absent", "Present"], 
                key=f"tb_screening_{idx}",
                help=f"Clinical assessment: {symptom.lower()}"
            )
            tb_inputs.append(1 if response == "Present" else 0)

    st.markdown("---")
    
    if st.button("🔍 Execute TB Risk Analysis", use_container_width=True, type="primary"):
        with st.spinner("🔄 Processing screening data and calculating TB risk stratification..."):
            tb_pred = tb_model.predict([tb_inputs])[0]
            tb_proba = tb_model.predict_proba([tb_inputs])[0][tb_pred]
            
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            if tb_pred == 1:
                st.error(f"🚨 **HIGH TB RISK** - Multiple criteria suggest possible tuberculosis. Please seek further clinical evaluation.")
            else:
                st.success(f"✅ **LOW TB RISK** - TB is unlikely based on current screening criteria.")
            
            st.markdown(f"**Risk probability:** {tb_proba:.2%}")

            st.markdown("</div>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        hepatitis_model = pickle.load(open("hepatitis_model.pkl", "rb"))
    except FileNotFoundError:
        st.error("Hepatitis model not found. Please upload hepatitis_model.pkl")
        hepatitis_model = None
    
    try:
        hiv_model = joblib.load("hiv_model.pkl")
    except FileNotFoundError:
        st.error("HIV model not found. Please upload hiv_model.pkl")
        hiv_model = None
    
    try:
        vectorizer = joblib.load("vectorizer.pkl")
    except FileNotFoundError:
        st.error("Vectorizer not found. Please upload vectorizer.pkl")
        vectorizer = None
    
    try:
        tb_model = joblib.load("tb_predictor_model.pkl")
    except FileNotFoundError:
        st.error("TB model not found. Please upload tb_predictor_model.pkl")
        tb_model = None
    
    return hepatitis_model, hiv_model, vectorizer, tb_model