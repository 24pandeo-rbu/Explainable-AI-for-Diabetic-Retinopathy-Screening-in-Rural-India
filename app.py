"""
EYE-CONIQ v2.0 — Explainable AI for Diabetic Retinopathy Screening
Main Streamlit Application

Features:
- Multi-language support (13 Indian languages)
- Blur detection with retake prompt
- DR severity grading with medical explanations
- Grad-CAM explainability
- Retinal structure segmentation
- Hospital finder with interactive map
- Ambulance calling & emergency contacts
- PDF clinical report generation
- Pipeline simulation dashboard
"""

import streamlit as st
import torch
import numpy as np
import cv2
import io
import datetime
from PIL import Image
from torchvision import transforms

# ── Page Config (must be first Streamlit call) ──
st.set_page_config(
    page_title="EYE-CONIQ | AI DR Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #00D4AA;
    --accent-blue: #4A9EFF;
    --accent-purple: #6C5CE7;
    --danger: #FF4757;
    --warning: #FFA502;
    --success: #2ED573;
    --bg-dark: #0A0F1C;
    --bg-card: #111827;
    --text-primary: #E8E8F0;
    --text-secondary: #8B8FA3;
    --border: rgba(255, 255, 255, 0.06);
}

.stApp { background: var(--bg-dark); font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 2rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1224 0%, #111827 100%); border-right: 1px solid var(--border); }
h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif !important; color: var(--text-primary) !important; }

.glass-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.9), rgba(26, 34, 53, 0.8));
    border: 1px solid var(--border); border-radius: 16px; padding: 24px;
    backdrop-filter: blur(20px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}
.glass-card:hover { border-color: rgba(0, 212, 170, 0.2); box-shadow: 0 8px 32px rgba(0, 212, 170, 0.08); }

.hero-title {
    font-size: 3.2rem; font-weight: 900;
    background: linear-gradient(135deg, #00D4AA, #4A9EFF, #6C5CE7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -1px; margin-bottom: 0; line-height: 1.1;
}
.hero-subtitle { font-size: 1.15rem; color: var(--text-secondary); font-weight: 300; margin-top: 4px; }

.severity-badge { display: inline-block; padding: 8px 20px; border-radius: 24px; font-weight: 700; font-size: 1.1rem; }
.severity-0 { background: linear-gradient(135deg, #2ED573, #1abc9c); color: #fff; }
.severity-1 { background: linear-gradient(135deg, #F1C40F, #f39c12); color: #333; }
.severity-2 { background: linear-gradient(135deg, #F39C12, #e67e22); color: #fff; }
.severity-3 { background: linear-gradient(135deg, #E74C3C, #c0392b); color: #fff; }
.severity-4 { background: linear-gradient(135deg, #C0392B, #962d22); color: #fff; animation: pulse-danger 1.5s ease-in-out infinite; }

@keyframes pulse-danger { 0%,100% { box-shadow: 0 0 15px rgba(231,76,60,0.3); } 50% { box-shadow: 0 0 30px rgba(231,76,60,0.6); } }
@keyframes pulse-glow { 0%,100% { box-shadow: 0 0 15px rgba(0,212,170,0.2); } 50% { box-shadow: 0 0 25px rgba(0,212,170,0.4); } }

.metric-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(26, 34, 53, 0.9));
    border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px;
    text-align: center; transition: all 0.3s ease;
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(0, 212, 170, 0.3); }
.metric-value { font-size: 1.8rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

.quality-bar { height: 8px; border-radius: 4px; background: rgba(255,255,255,0.06); overflow: hidden; margin: 4px 0; }
.quality-fill { height: 100%; border-radius: 4px; transition: width 1s ease; }

.section-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,212,170,0.3), transparent); margin: 24px 0; }

.blur-warning {
    background: linear-gradient(135deg, rgba(255,165,2,0.15), rgba(255,71,87,0.1));
    border: 2px solid rgba(255,165,2,0.5); border-left: 5px solid #FFA502;
    border-radius: 16px; padding: 24px; animation: pulse-glow 2s ease-in-out infinite;
}

.emergency-banner {
    background: linear-gradient(135deg, rgba(231,76,60,0.2), rgba(192,57,43,0.15));
    border: 2px solid rgba(231,76,60,0.5); border-radius: 16px; padding: 20px;
    animation: pulse-danger 2s ease-in-out infinite;
}

.ambulance-btn {
    display: inline-block; background: linear-gradient(135deg, #FF4757, #c0392b);
    color: white !important; padding: 12px 28px; border-radius: 12px;
    font-weight: 800; font-size: 1.1rem; text-decoration: none;
    box-shadow: 0 4px 15px rgba(255,71,87,0.4); transition: all 0.3s ease;
}
.ambulance-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(255,71,87,0.5); }

.hospital-btn {
    display: inline-block; background: linear-gradient(135deg, #4A9EFF, #2d7dd2);
    color: white !important; padding: 10px 24px; border-radius: 10px;
    font-weight: 700; font-size: 0.95rem; text-decoration: none;
}

.referable-warning {
    background: linear-gradient(135deg, rgba(231,76,60,0.15), rgba(192,57,43,0.1));
    border: 1px solid rgba(231,76,60,0.4); border-left: 4px solid #E74C3C;
    border-radius: 12px; padding: 16px 20px;
}

.do-item { color: #2ED573; } .dont-item { color: #FF4757; }

.scan-line { position: relative; overflow: hidden; }
.scan-line::after { content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%; background: linear-gradient(90deg, transparent, rgba(0,212,170,0.05), transparent); animation: scan 3s ease-in-out infinite; }
@keyframes scan { 0% { left: -100%; } 100% { left: 100%; } }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: rgba(17,24,39,0.5); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; font-weight: 600; color: var(--text-secondary); }
.stTabs [aria-selected="true"] { background: rgba(0,212,170,0.15) !important; color: var(--primary) !important; }

.stButton > button { background: linear-gradient(135deg, #00D4AA, #00b894) !important; color: #0A0F1C !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; font-family: 'Inter', sans-serif !important; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(0,212,170,0.3) !important; }
.stDownloadButton > button { background: linear-gradient(135deg, #4A9EFF, #2d7dd2) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; }
.stFileUploader { border: 2px dashed rgba(0,212,170,0.3) !important; border-radius: 16px !important; background: rgba(0,212,170,0.03) !important; }

.legend-item { display: inline-flex; align-items: center; gap: 6px; margin-right: 16px; font-size: 0.85rem; color: var(--text-secondary); }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Import project modules ──
from model import load_trained_model
from image_quality import compute_quality_report, enhance_fundus_image, ImageGrade
from explainability import (
    GradCAM, create_heatmap_overlay, annotate_with_evidence,
    generate_clinical_evidence, DR_CLINICAL_CRITERIA
)
from retinal_analysis import run_full_segmentation, get_lesion_counts
from report_generator import generate_pdf_report
from simulink_model import PipelineConfig, simulate_pipeline
from translations import t, LANGUAGES
from medical_knowledge import get_medical_info
from hospital_finder import (
    get_hospitals_by_state, get_all_states, create_hospital_map,
    EMERGENCY_NUMBERS
)
from streamlit_folium import st_folium

# ── Class Names ──
CLASS_NAMES = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
severity_colors = {0: "#2ED573", 1: "#F1C40F", 2: "#F39C12", 3: "#E74C3C", 4: "#C0392B"}

# ── Model Loading ──
@st.cache_resource
def get_gradcam_model():
    device = torch.device("cpu")
    model = load_trained_model("eye_coniq_model.pth", device)
    gradcam = GradCAM(model, target_layer="layer4")
    return gradcam, model, device

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 16px 0 8px 0;">
        <div style="font-size: 2.5rem;">👁️</div>
        <div class="hero-title" style="font-size: 1.8rem;">EYE-CONIQ</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Language selector
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    lang_options = list(LANGUAGES.values())
    lang_codes = list(LANGUAGES.keys())
    selected_lang_idx = st.selectbox(
        "🌐 Language / भाषा / மொழி",
        range(len(lang_options)),
        format_func=lambda i: lang_options[i],
        index=0,
    )
    lang = lang_codes[selected_lang_idx]
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        [t("nav_screening", lang), t("nav_hospitals", lang), t("nav_simulator", lang), t("nav_about", lang)],
        label_visibility="collapsed",
    )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Emergency buttons in sidebar
    st.markdown(f"""
    <div style="padding: 8px 0;">
        <a href="tel:108" class="ambulance-btn" style="display: block; text-align: center; margin-bottom: 8px; font-size: 0.95rem; padding: 10px;">
            {t("call_ambulance", lang)}
        </a>
        <a href="tel:112" class="hospital-btn" style="display: block; text-align: center;">
            {t("emergency_helpline", lang)}
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 0.7rem; color: #444455; padding: 4px 0;">
        ⚕️ <i>For clinical screening support only.<br>Not a substitute for professional diagnosis.</i>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE: SCREENING
# ──────────────────────────────────────────────
if page == t("nav_screening", lang):
    st.markdown(f"""
    <div style="text-align: center; padding: 0 0 20px 0;">
        <h1 class="hero-title">{t("app_subtitle", lang)}</h1>
        <p class="hero-subtitle">{t("upload_hint", lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        t("upload_title", lang),
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
    )
    
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("❌ Invalid image file.")
            st.stop()
        
        image_rgb = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # ── STEP 1: Quality Assessment & Blur Detection ──
        quality_report = compute_quality_report(image_bgr)
        
        # ── BLUR GATE: Check if image is too blurry ──
        if quality_report.focus_score < 30 or quality_report.grade == ImageGrade.REJECT:
            st.markdown(f"""
            <div class="blur-warning">
                <div style="font-size: 1.5rem; font-weight: 800; color: #FFA502; margin-bottom: 12px;">
                    {t("image_blurry_title", lang)}
                </div>
                <div style="color: var(--text-primary); font-size: 1rem; margin-bottom: 16px;">
                    {t("image_blurry_msg", lang)}
                </div>
                <div style="color: var(--text-secondary); line-height: 2; font-size: 0.95rem;">
                    {t("blur_tip_1", lang)}<br>
                    {t("blur_tip_2", lang)}<br>
                    {t("blur_tip_3", lang)}
                </div>
                <div style="margin-top: 16px; padding: 12px; background: rgba(255,71,87,0.1); border-radius: 10px;">
                    <div style="color: #FF4757; font-weight: 700; font-size: 1.1rem;">
                        {t("retake_photo", lang)}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show the blurry image for reference
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image_rgb, caption="Uploaded image (blurry)", use_container_width=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="border-color: #FF4757;">
                    <div class="metric-value" style="color: #FF4757;">{quality_report.focus_score:.0f}</div>
                    <div class="metric-label">Focus Score</div>
                    <div class="quality-bar"><div class="quality-fill" style="width: {quality_report.focus_score}%; background: #FF4757;"></div></div>
                </div>
                """, unsafe_allow_html=True)
            
            # Option to continue anyway
            st.markdown("")
            continue_anyway = st.checkbox(t("continue_anyway", lang), value=False)
            
            if not continue_anyway:
                st.stop()
        
        # ── STEP 2: Full Analysis Pipeline ──
        with st.status(t("processing", lang), expanded=True) as status:
            st.write("📋 Assessing image quality...")
            enhanced_bgr = enhance_fundus_image(image_bgr, quality_report)
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            
            st.write("🧠 Running DR classification with Grad-CAM...")
            gradcam_engine, model, device = get_gradcam_model()
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            cam, predicted_class, probs = gradcam_engine.generate(input_tensor)
            confidence = probs[predicted_class].item() * 100
            
            gradcam_overlay = create_heatmap_overlay(image_bgr, cam, alpha=0.45)
            annotated_overlay = annotate_with_evidence(image_bgr, cam, predicted_class, confidence)
            clinical_evidence = generate_clinical_evidence(cam, predicted_class, probs)
            
            st.write("🔬 Segmenting retinal structures...")
            seg_result = run_full_segmentation(image_bgr)
            lesion_counts = get_lesion_counts(seg_result)
            
            status.update(label=t("analysis_complete", lang), state="complete")
        
        # Get medical info in selected language
        medical_info = get_medical_info(predicted_class, lang)
        criteria = DR_CLINICAL_CRITERIA[predicted_class]
        sev_color = severity_colors[predicted_class]
        severity_name = t(f"severity_{predicted_class}", lang)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # ── DIAGNOSIS BANNER ──
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-color: {sev_color}40;">
            <div style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">
                {t("ai_diagnosis", lang)}
            </div>
            <div class="severity-badge severity-{predicted_class}">{severity_name}</div>
            <div style="margin-top: 12px;">
                <span class="metric-value" style="color: {sev_color};">{confidence:.1f}%</span>
                <span style="color: var(--text-secondary); font-size: 0.9rem;"> {t("confidence", lang)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ── URGENCY BANNER (for severe/critical cases) ──
        if medical_info["urgency"] in ("high", "critical"):
            st.markdown(f"""
            <div class="emergency-banner" style="margin-top: 16px;">
                <div style="font-size: 1.2rem; font-weight: 800; color: #FF4757; margin-bottom: 8px;">
                    {medical_info["urgency_message"]}
                </div>
                <div style="display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap;">
                    <a href="tel:108" class="ambulance-btn">{t("call_ambulance", lang)}</a>
                    <a href="tel:112" class="hospital-btn">{t("emergency_helpline", lang)}</a>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif medical_info["urgency"] == "medium":
            st.markdown(f"""
            <div class="referable-warning" style="margin-top: 16px;">
                <div style="font-size: 1rem; font-weight: 700; color: #FFA502;">
                    {medical_info["urgency_message"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # ── ANALYSIS TABS ──
        tab_labels = [
            t("tab_explanation", lang), t("tab_images", lang), t("tab_gradcam", lang),
            t("tab_segmentation", lang), t("tab_hospitals", lang), t("tab_report", lang),
        ]
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_labels)
        
        # ──── TAB 1: EXPLANATION & CURE (NEW - the main new feature) ────
        with tab1:
            st.markdown(f"##### 🔍 {t('tab_explanation', lang)}")
            
            # What is the problem?
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 16px;">
                <h4 style="color: #4A9EFF; margin-top: 0;">🔍 What is this condition?</h4>
                <div style="color: var(--text-primary); line-height: 1.8; font-size: 1rem;">
                    {medical_info["what_is"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Why did this happen?
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 16px;">
                <h4 style="color: #FFA502; margin-top: 0;">⚡ Why did this happen? (Causes)</h4>
                <div style="color: var(--text-primary); line-height: 1.8; font-size: 1rem;">
                    {medical_info["causes"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Symptoms
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 16px;">
                <h4 style="color: #6C5CE7; margin-top: 0;">👁️ Symptoms</h4>
                <div style="color: var(--text-primary); line-height: 1.8; font-size: 1rem;">
                    {medical_info["symptoms"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Treatment / How to Cure
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 16px; border-color: rgba(0,212,170,0.3);">
                <h4 style="color: #00D4AA; margin-top: 0;">💊 Treatment & How to Cure</h4>
                <div style="color: var(--text-primary); line-height: 1.8; font-size: 1rem;">
                    {medical_info["treatment"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Lifestyle changes
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 16px;">
                <h4 style="color: #2ED573; margin-top: 0;">🏃 Lifestyle Changes</h4>
                <div style="color: var(--text-primary); line-height: 2.2; font-size: 1rem;">
                    {"<br>".join(medical_info["lifestyle"])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Do's and Don'ts
            do_col, dont_col = st.columns(2)
            with do_col:
                do_items = "".join(f'<div class="do-item" style="margin: 6px 0;">✅ {item}</div>' for item in medical_info["do_list"])
                st.markdown(f"""
                <div class="glass-card" style="border-color: rgba(46,213,115,0.3);">
                    <h4 style="color: #2ED573; margin-top: 0;">✅ DO's</h4>
                    <div style="font-size: 0.95rem; line-height: 1.8;">{do_items}</div>
                </div>
                """, unsafe_allow_html=True)
            with dont_col:
                dont_items = "".join(f'<div class="dont-item" style="margin: 6px 0;">❌ {item}</div>' for item in medical_info["dont_list"])
                st.markdown(f"""
                <div class="glass-card" style="border-color: rgba(255,71,87,0.3);">
                    <h4 style="color: #FF4757; margin-top: 0;">❌ DON'Ts</h4>
                    <div style="font-size: 0.95rem; line-height: 1.8;">{dont_items}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Emergency section for severe cases
            if predicted_class >= 3:
                st.markdown(f"""
                <div class="emergency-banner" style="margin-top: 16px;">
                    <div style="font-size: 1.3rem; font-weight: 800; color: #FF4757; margin-bottom: 12px;">
                        🚨 {"IMMEDIATE MEDICAL ATTENTION REQUIRED" if lang == "en" else medical_info["urgency_message"]}
                    </div>
                    <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                        <a href="tel:108" class="ambulance-btn">{t("call_ambulance", lang)}</a>
                        <a href="tel:112" class="hospital-btn">{t("emergency_helpline", lang)}</a>
                    </div>
                    <div style="color: var(--text-secondary); margin-top: 12px; font-size: 0.85rem;">
                        📞 National Eye Helpline: 1800-345-4545 (Toll Free)
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ──── TAB 2: IMAGES ────
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Original Image")
                st.image(image_rgb, use_container_width=True)
            with col2:
                st.markdown("##### Enhanced Image")
                st.image(enhanced_rgb, use_container_width=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("##### 📋 Image Quality Assessment")
            
            grade_emoji = {"accept": "✅", "borderline": "⚠️", "reject": "❌"}
            grade_color = {"accept": "#2ED573", "borderline": "#FFA502", "reject": "#FF4757"}
            grade = quality_report.grade.value
            
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-size: 1.2rem; font-weight: 700; color: {grade_color[grade]};">
                    {grade_emoji[grade]} {quality_report.feedback}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            qcols = st.columns(4)
            metrics = [
                ("Focus", quality_report.focus_score, "#4A9EFF"),
                ("Illumination", quality_report.illumination_score, "#FFA502"),
                ("Field of View", quality_report.fov_score, "#6C5CE7"),
                ("Contrast", quality_report.contrast_score, "#00D4AA"),
            ]
            for col, (name, score, color) in zip(qcols, metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {color};">{score:.0f}</div>
                        <div class="metric-label">{name}</div>
                        <div class="quality-bar"><div class="quality-fill" style="width: {score}%; background: {color};"></div></div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # ──── TAB 3: GRAD-CAM ────
        with tab3:
            st.markdown("##### 🔥 Grad-CAM Explainability")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Grad-CAM Heatmap**")
                st.image(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.markdown("**Evidence Map**")
                st.image(cv2.cvtColor(annotated_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Spatial analysis
            quadrant_data = clinical_evidence.get("quadrant_activations", {})
            qcols = st.columns(4)
            quad_colors = ["#4A9EFF", "#6C5CE7", "#00D4AA", "#FFA502"]
            most_affected = clinical_evidence.get("most_affected_quadrant", "")
            for i, (col, (qname, qval)) in enumerate(zip(qcols, quadrant_data.items())):
                with col:
                    is_max = qname == most_affected
                    border = f"border: 2px solid {quad_colors[i]};" if is_max else ""
                    st.markdown(f"""
                    <div class="metric-card" style="{border}">
                        <div class="metric-value" style="color: {quad_colors[i]};">{qval:.3f}</div>
                        <div class="metric-label">{qname}</div>
                        {'<div style="color: #00D4AA; font-size: 0.7rem; margin-top: 4px;">▲ HIGHEST</div>' if is_max else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Probability breakdown
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("##### 📊 Class Probabilities")
            for cls_idx in range(5):
                prob_val = probs[cls_idx].item() * 100
                is_pred = cls_idx == predicted_class
                st.markdown(f"""
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                        <span style="color: {'#E8E8F0' if is_pred else '#8B8FA3'}; font-weight: {'700' if is_pred else '400'};">
                            {CLASS_NAMES[cls_idx]}{'  ◄' if is_pred else ''}
                        </span>
                        <span style="color: {severity_colors[cls_idx]}; font-family: 'JetBrains Mono'; font-weight: 600;">{prob_val:.1f}%</span>
                    </div>
                    <div class="quality-bar"><div class="quality-fill" style="width: {prob_val}%; background: {severity_colors[cls_idx]};"></div></div>
                </div>
                """, unsafe_allow_html=True)
        
        # ──── TAB 4: SEGMENTATION ────
        with tab4:
            st.markdown("##### 🔬 Retinal Structure Analysis")
            st.markdown("""
            <div style="margin-bottom: 12px;">
                <span class="legend-item"><span class="legend-dot" style="background: #FF0000;"></span> Blood Vessels</span>
                <span class="legend-item"><span class="legend-dot" style="background: #FFFF00;"></span> Exudates</span>
                <span class="legend-item"><span class="legend-dot" style="background: #0000FF;"></span> Hemorrhages</span>
                <span class="legend-item"><span class="legend-dot" style="background: #00FFFF;"></span> Microaneurysms</span>
                <span class="legend-item"><span class="legend-dot" style="background: #00FF00;"></span> Optic Disc</span>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(seg_result.composite_overlay, cv2.COLOR_BGR2RGB), caption="Structure Overlay", use_container_width=True)
            with col2:
                vessel_display = np.zeros_like(image_rgb)
                vessel_display[:, :, 0] = seg_result.vessels_mask
                st.image(vessel_display, caption="Vessel Map", use_container_width=True)
            
            lcols = st.columns(5)
            lesion_metrics = [
                ("MAs", lesion_counts["microaneurysm_count"], "#00FFFF"),
                ("Exudates", lesion_counts["exudate_count"], "#F1C40F"),
                ("Hemorrhages", lesion_counts["hemorrhage_count"], "#4A9EFF"),
                ("Vessel %", f"{lesion_counts['vessel_density_pct']:.1f}", "#FF4757"),
                ("Optic Disc", "✅" if lesion_counts["optic_disc_detected"] else "❌", "#2ED573"),
            ]
            for col, (name, value, color) in zip(lcols, lesion_metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {color}; font-size: 1.5rem;">{value}</div>
                        <div class="metric-label">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # ──── TAB 5: HOSPITALS & EMERGENCY ────
        with tab5:
            st.markdown(f"##### {t('tab_hospitals', lang)}")
            
            # Emergency buttons at top
            if predicted_class >= 2:
                st.markdown(f"""
                <div class="emergency-banner" style="margin-bottom: 16px;">
                    <div style="font-size: 1.1rem; font-weight: 800; color: #FF4757; margin-bottom: 8px;">
                        🚨 {"Urgent: Find an eye specialist near you!" if predicted_class >= 3 else "Recommended: Consult an eye specialist"}
                    </div>
                    <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                        <a href="tel:108" class="ambulance-btn">{t("call_ambulance", lang)}</a>
                        <a href="tel:112" class="hospital-btn">{t("emergency_helpline", lang)}</a>
                        <a href="tel:18003454545" class="hospital-btn" style="background: linear-gradient(135deg, #6C5CE7, #5a4bd1);">
                            📞 Eye Helpline: 1800-345-4545
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # State selector
            all_states = get_all_states()
            selected_state = st.selectbox(
                t("select_state", lang),
                ["All India"] + all_states,
                index=0,
            )
            
            # Get hospitals
            if selected_state == "All India":
                hospitals = get_hospitals_by_state(None) if selected_state == "All India" else get_hospitals_by_state(selected_state)
                from hospital_finder import HOSPITALS
                hospitals = HOSPITALS
            else:
                hospitals = get_hospitals_by_state(selected_state)
            
            if hospitals:
                # Create and display map
                hospital_map = create_hospital_map(hospitals)
                st_folium(hospital_map, width=None, height=450, use_container_width=True)
                
                # Hospital list
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown(f"##### 📋 {len(hospitals)} Eye Hospitals Found")
                
                for hosp in hospitals:
                    emergency_badge = '<span style="background: #FF4757; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 700;">🚨 24/7 EMERGENCY</span>' if hosp.get("emergency") else ""
                    type_color = {"Government": "#4A9EFF", "Trust": "#2ED573", "Private": "#FFA502"}.get(hosp.get("type", ""), "#888")
                    specialists = " • ".join(hosp.get("specialists", []))
                    
                    st.markdown(f"""
                    <div class="glass-card" style="margin-bottom: 12px; padding: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 8px;">
                            <div>
                                <div style="font-weight: 700; color: var(--text-primary); font-size: 1.05rem;">
                                    🏥 {hosp['name']} {emergency_badge}
                                </div>
                                <div style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 4px;">
                                    📍 {hosp['address']}
                                </div>
                                <div style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 2px;">
                                    👨‍⚕️ {specialists}
                                </div>
                                <div style="margin-top: 4px;">
                                    <span style="color: {type_color}; font-size: 0.8rem; font-weight: 600;">{hosp.get('type', '')}</span>
                                </div>
                            </div>
                            <div style="display: flex; gap: 8px; align-items: center;">
                                <a href="tel:{hosp['phone']}" class="hospital-btn" style="font-size: 0.85rem; padding: 8px 16px;">
                                    📞 {hosp['phone']}
                                </a>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No hospitals found in {selected_state}. Try selecting 'All India'.")
        
        # ──── TAB 6: REPORT ────
        with tab6:
            st.markdown(f"##### {t('tab_report', lang)}")
            
            session_id = f"EYC-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            quality_dict = {
                "focus_score": quality_report.focus_score,
                "illumination_score": quality_report.illumination_score,
                "fov_score": quality_report.fov_score,
                "contrast_score": quality_report.contrast_score,
                "overall_score": quality_report.overall_score,
                "grade": quality_report.grade.value.title(),
            }
            
            report_cols = st.columns(2)
            with report_cols[0]:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-weight: 700; color: #4A9EFF; margin-bottom: 8px;">Report Contents</div>
                    <div style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8;">
                        ✅ Original & Enhanced images<br>✅ Grad-CAM attention heatmap<br>
                        ✅ Retinal structure segmentation<br>✅ DR severity grade & confidence<br>
                        ✅ Image quality metrics<br>✅ Lesion analysis<br>
                        ✅ Clinical recommendation<br>✅ Disclaimer & metadata
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with report_cols[1]:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-weight: 700; color: #00D4AA; margin-bottom: 8px;">Session Summary</div>
                    <div style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8;">
                        <b style="color: #E8E8F0;">Diagnosis:</b> {severity_name}<br>
                        <b style="color: #E8E8F0;">Confidence:</b> {confidence:.1f}%<br>
                        <b style="color: #E8E8F0;">Quality:</b> {quality_report.grade.value.title()}<br>
                        <b style="color: #E8E8F0;">Referable DR:</b> {'Yes ⚠️' if clinical_evidence.get('is_referable') else 'No ✅'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            try:
                pdf_bytes = generate_pdf_report(
                    original_image=image_bgr, enhanced_image=enhanced_bgr,
                    gradcam_overlay=gradcam_overlay, segmentation_overlay=seg_result.composite_overlay,
                    clinical_evidence=clinical_evidence, quality_report=quality_dict,
                    lesion_counts=lesion_counts, session_id=session_id,
                )
                st.download_button(
                    label=t("download_report", lang), data=pdf_bytes,
                    file_name=f"EYE-CONIQ_Report_{session_id}.pdf",
                    mime="application/pdf", type="primary", use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
    
    else:
        # Landing state
        st.markdown(f"""
        <div class="glass-card scan-line" style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 4rem; margin-bottom: 16px;">🔬</div>
            <div style="font-size: 1.3rem; font-weight: 600; color: var(--text-primary);">{t("upload_title", lang)}</div>
            <div style="color: var(--text-secondary); max-width: 500px; margin: 8px auto 0; line-height: 1.6;">
                {t("upload_hint", lang)}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        feat_cols = st.columns(4)
        features = [
            ("📋", "Quality Check", "#4A9EFF"), ("🧠", "AI Grading", "#00D4AA"),
            ("🔥", "Grad-CAM XAI", "#6C5CE7"), ("🏥", "Hospital Finder", "#FFA502"),
        ]
        for col, (icon, title, color) in zip(feat_cols, features):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="padding: 20px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
                    <div style="font-weight: 700; color: {color};">{title}</div>
                </div>
                """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE: HOSPITALS
# ──────────────────────────────────────────────
elif page == t("nav_hospitals", lang):
    st.markdown(f"""
    <div style="text-align: center; padding: 0 0 20px 0;">
        <h1 class="hero-title">{t("find_hospitals", lang)}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency banner
    st.markdown(f"""
    <div class="emergency-banner" style="margin-bottom: 20px;">
        <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; align-items: center;">
            <a href="tel:108" class="ambulance-btn">{t("call_ambulance", lang)}</a>
            <a href="tel:112" class="hospital-btn">{t("emergency_helpline", lang)}</a>
            <a href="tel:18003454545" class="hospital-btn" style="background: linear-gradient(135deg, #6C5CE7, #5a4bd1);">
                📞 Eye Helpline: 1800-345-4545
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    selected_state = st.selectbox(t("select_state", lang), ["All India"] + get_all_states())
    
    if selected_state == "All India":
        from hospital_finder import HOSPITALS
        hospitals = HOSPITALS
    else:
        hospitals = get_hospitals_by_state(selected_state)
    
    if hospitals:
        hospital_map = create_hospital_map(hospitals)
        st_folium(hospital_map, width=None, height=500, use_container_width=True)
        
        st.markdown(f"##### 📋 {len(hospitals)} Eye Hospitals")
        for hosp in hospitals:
            emergency_badge = '<span style="background: #FF4757; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 700;">🚨 24/7</span>' if hosp.get("emergency") else ""
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 10px; padding: 14px;">
                <div style="font-weight: 700; color: var(--text-primary);">🏥 {hosp['name']} {emergency_badge}</div>
                <div style="color: var(--text-secondary); font-size: 0.85rem;">📍 {hosp['address']}</div>
                <div style="margin-top: 6px;">
                    <a href="tel:{hosp['phone']}" class="hospital-btn" style="font-size: 0.8rem; padding: 6px 14px;">📞 {hosp['phone']}</a>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE: PIPELINE SIMULATOR
# ──────────────────────────────────────────────
elif page == t("nav_simulator", lang):
    st.markdown("""
    <div style="text-align: center; padding: 0 0 20px 0;">
        <h1 class="hero-title">Screening Pipeline Simulator</h1>
        <p class="hero-subtitle">Monte Carlo simulation for district-level DR screening programs</p>
    </div>
    """, unsafe_allow_html=True)
    
    cfg_cols = st.columns(4)
    with cfg_cols[0]: num_cameras = st.slider("📷 Cameras", 1, 20, 5)
    with cfg_cols[1]: num_servers = st.slider("🖥️ Servers", 1, 5, 1)
    with cfg_cols[2]: num_doctors = st.slider("👨‍⚕️ Doctors", 1, 8, 2)
    with cfg_cols[3]: bandwidth = st.selectbox("📡 Bandwidth", ["2G", "3G", "4G", "WiFi"], index=1)
    
    target_pop = st.slider("🎯 Target Population", 10000, 500000, 100000, step=10000)
    
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running simulation..."):
            config = PipelineConfig(total_patients=target_pop, num_fundus_cameras=num_cameras,
                                    num_gpu_servers=num_servers, num_ophthalmologists=num_doctors, bandwidth_type=bandwidth)
            sim = simulate_pipeline(config, num_simulations=100)
        
        mcols = st.columns(5)
        sim_metrics = [
            ("Throughput", f"{sim['annual_throughput']:,}", "#00D4AA"),
            ("Target Met", f"{sim['target_met_pct']:.1f}%", "#4A9EFF" if sim['target_met_pct'] >= 80 else "#FF4757"),
            ("Daily Avg", f"{sim['daily_throughput_mean']:.0f}", "#6C5CE7"),
            ("Wait Time", f"{sim['avg_wait_minutes']:.1f}m", "#FFA502"),
            ("Cost/Screen", f"₹{sim['cost_per_screening_inr']:.0f}", "#2ED573"),
        ]
        for col, (label, val, color) in zip(mcols, sim_metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {color}; font-size: 1.5rem;">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div style="color: #FF4757; font-weight: 700;">🚧 Primary Bottleneck: {sim['primary_bottleneck']}</div>
            <div style="color: var(--text-secondary); margin-top: 8px;">Annual Cost: ₹{sim['annual_cost_inr']/100000:.1f} Lakhs</div>
        </div>
        """, unsafe_allow_html=True)
        
        if sim.get("bottleneck_distribution"):
            st.bar_chart(sim["bottleneck_distribution"], color="#FF4757")


# ──────────────────────────────────────────────
# PAGE: ABOUT
# ──────────────────────────────────────────────
elif page == t("nav_about", lang):
    st.markdown(f"""
    <div style="text-align: center; padding: 0 0 30px 0;">
        <div style="font-size: 4rem;">👁️</div>
        <h1 class="hero-title">EYE-CONIQ</h1>
        <p class="hero-subtitle">{t("app_subtitle", lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #4A9EFF; margin-top: 0;">🏥 The Problem</h4>
        <div style="color: var(--text-secondary); line-height: 1.8;">
            India has over <b style="color: #E8E8F0;">77 million diabetic adults</b> — the second highest globally.
            Diabetic Retinopathy affects ~18% of this population and is a leading cause of
            <b style="color: #FF4757;">preventable blindness</b>. Early screening can prevent 90% of vision loss.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    feat_cols = st.columns(3)
    features = [
        ("📋", "Quality Check", "Auto blur detection with retake prompts", "#4A9EFF"),
        ("🧠", "AI + Explainability", "5-level DR grading with Grad-CAM", "#00D4AA"),
        ("💊", "Medical Guidance", "Causes, treatment & lifestyle advice", "#6C5CE7"),
        ("🏥", "Hospital Finder", "Nearest eye hospitals on interactive map", "#FFA502"),
        ("🚑", "Emergency", "Ambulance calling & helpline numbers", "#FF4757"),
        ("🌐", "Multi-Language", "13 Indian languages for rural accessibility", "#2ED573"),
    ]
    for i, (icon, title, desc, color) in enumerate(features):
        with feat_cols[i % 3]:
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 16px; min-height: 150px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
                <div style="font-weight: 700; color: {color};">{title}</div>
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)