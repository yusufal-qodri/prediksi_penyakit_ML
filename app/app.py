import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="DeteksiPenyakit.id",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Instrument+Serif:ital@0;1&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #f0f4f8; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f2942 0%, #1a3f6f 60%, #1e5799 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e8f1fb !important; }
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.07);
    border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;
    display: block; transition: all 0.2s;
    border: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.15); }

.hero {
    background: linear-gradient(135deg, #0f2942 0%, #1a5276 50%, #1e8bc3 100%);
    border-radius: 20px; padding: 36px 40px; margin-bottom: 28px;
    color: white; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero h1 {
    font-family: 'Instrument Serif', serif; font-size: 2.2rem;
    font-weight: 400; margin: 0 0 8px 0; line-height: 1.2;
}
.hero p { font-size: 0.95rem; opacity: 0.8; margin: 0; font-weight: 300; }

.card {
    background: white; border-radius: 16px; padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 16px;
    border: 1px solid rgba(0,0,0,0.05);
}

.result-card {
    background: white; border-radius: 14px; padding: 20px;
    text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid #1a5276; transition: transform 0.2s;
}
.result-card:hover { transform: translateY(-2px); }
.result-card.gold   { border-top-color: #f59e0b; }
.result-card.silver { border-top-color: #94a3b8; }
.result-card.bronze { border-top-color: #b45309; }
.result-medal { font-size: 2rem; margin-bottom: 4px; }
.result-name  { font-size: 1rem; font-weight: 700; color: #1e293b; margin-bottom: 8px; }
.result-pct   { font-size: 2rem; font-weight: 800; color: #1a5276; }

.info-box {
    background: #f0f7ff; border-left: 4px solid #1a5276;
    border-radius: 0 12px 12px 0; padding: 16px 20px; margin-bottom: 12px;
    font-size: 0.95rem; color: #1e293b; line-height: 1.6;
}
.precaution-item {
    background: #f0fdf4; border-left: 4px solid #22c55e;
    border-radius: 0 12px 12px 0; padding: 12px 16px; margin-bottom: 10px;
    font-size: 0.92rem; color: #166534; display: flex; align-items: center; gap: 10px;
}
.prevention-item {
    background: #fdf4ff; border-left: 4px solid #a855f7;
    border-radius: 0 12px 12px 0; padding: 12px 16px; margin-bottom: 10px;
    font-size: 0.92rem; color: #6b21a8; display: flex; align-items: center; gap: 10px;
}
.doctor-box {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #bfdbfe; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 12px;
    display: flex; align-items: center; gap: 14px;
}

.severity-bar-wrap {
    background: #e2e8f0; border-radius: 999px;
    height: 12px; width: 100%; margin: 10px 0;
    overflow: hidden;
}
.severity-bar {
    height: 100%; border-radius: 999px;
    transition: width 0.6s ease;
}

.stat-card {
    background: white; border-radius: 14px; padding: 20px 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border: 1px solid #e8eef5; text-align: center;
}
.stat-number { font-size: 2.2rem; font-weight: 800; color: #1a5276; line-height: 1; margin-bottom: 6px; }
.stat-label  { font-size: 0.85rem; color: #64748b; font-weight: 500; }

.section-title {
    font-size: 1.05rem; font-weight: 700; color: #0f2942;
    margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #d1dbe8, transparent);
    margin: 24px 0;
}
.history-item {
    background: white; border-radius: 12px; padding: 14px 18px;
    margin-bottom: 10px; border: 1px solid #e2e8f0;
    display: flex; justify-content: space-between; align-items: center;
}
.tag {
    background: #dbeafe; color: #1e40af; border-radius: 999px;
    padding: 3px 12px; font-size: 0.78rem; font-weight: 600;
}
.ens-card {
    background: white; border-radius: 14px; padding: 20px;
    border: 1px solid #e2e8f0; cursor: pointer;
    transition: all 0.2s; margin-bottom: 12px;
}
.ens-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); transform: translateY(-1px); }

.warning-box {
    background: #fffbeb; border: 1px solid #fcd34d; border-radius: 12px;
    padding: 14px 18px; color: #92400e; font-size: 0.9rem; margin-bottom: 16px;
}

.stButton > button {
    background: linear-gradient(135deg, #0f2942, #1a5276) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 14px 28px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(15,41,66,0.3) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Path setup ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load resources ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model     = joblib.load(os.path.join(BASE_DIR, 'model', 'model.pkl'))
    le        = joblib.load(os.path.join(BASE_DIR, 'model', 'label_encoder.pkl'))
    explainer = joblib.load(os.path.join(BASE_DIR, 'model', 'shap_explainer.pkl'))
    return model, le, explainer

@st.cache_data
def load_data():
    with open(os.path.join(BASE_DIR, 'data', 'symptoms_list.json')) as f:
        symptoms = json.load(f)
    with open(os.path.join(BASE_DIR, 'data', 'symptom_translation.json')) as f:
        translation = json.load(f)
    with open(os.path.join(BASE_DIR, 'data', 'doctor_mapping.json')) as f:
        doctor_map = json.load(f)
    with open(os.path.join(BASE_DIR, 'data', 'prevention_mapping.json')) as f:
        prevention_map = json.load(f)

    # Pakai versi terjemahan jika ada, fallback ke versi asli
    desc_path = os.path.join(BASE_DIR, 'data', 'symptom_Description_id.csv')
    if not os.path.exists(desc_path):
        desc_path = os.path.join(BASE_DIR, 'data', 'symptom_Description.csv')
    prec_path = os.path.join(BASE_DIR, 'data', 'symptom_precaution_id.csv')
    if not os.path.exists(prec_path):
        prec_path = os.path.join(BASE_DIR, 'data', 'symptom_precaution.csv')

    df_desc    = pd.read_csv(desc_path).set_index('Disease')
    df_prec    = pd.read_csv(prec_path).set_index('Disease')
    df_encoded = pd.read_csv(os.path.join(BASE_DIR, 'data', 'df_encoded.csv'))
    df_severity= pd.read_csv(os.path.join(BASE_DIR, 'data', 'Symptom-severity.csv'))
    df_severity.columns = df_severity.columns.str.strip()
    df_severity['Symptom'] = df_severity['Symptom'].str.strip()
    severity_dict = dict(zip(df_severity['Symptom'], df_severity['weight']))

    return symptoms, translation, doctor_map, prevention_map, df_desc, df_prec, df_encoded, severity_dict

@st.cache_data
def load_disease_translation():
    return {
        "Fungal infection": "Infeksi Jamur", "Allergy": "Alergi", "GERD": "GERD",
        "Chronic cholestasis": "Kolestasis Kronis", "Drug Reaction": "Reaksi Obat",
        "Peptic ulcer diseae": "Tukak Lambung", "AIDS": "AIDS", "Diabetes": "Diabetes",
        "Gastroenteritis": "Gastroenteritis", "Bronchial Asthma": "Asma Bronkial",
        "Hypertension": "Hipertensi", "Migraine": "Migrain",
        "Cervical spondylosis": "Spondilosis Serviks",
        "Paralysis (brain hemorrhage)": "Kelumpuhan (Perdarahan Otak)",
        "Jaundice": "Penyakit Kuning", "Malaria": "Malaria", "Chicken pox": "Cacar Air",
        "Dengue": "Demam Berdarah", "Typhoid": "Tifoid", "Hepatitis A": "Hepatitis A",
        "Hepatitis B": "Hepatitis B", "Hepatitis C": "Hepatitis C",
        "Hepatitis D": "Hepatitis D", "Hepatitis E": "Hepatitis E",
        "Alcoholic hepatitis": "Hepatitis Alkoholik", "Tuberculosis": "Tuberkulosis",
        "Common Cold": "Flu Biasa", "Pneumonia": "Pneumonia",
        "Dimorphic hemmorhoids(piles)": "Wasir", "Heart attack": "Serangan Jantung",
        "Variceal bleeding": "Perdarahan Varises", "Hypothyroidism": "Hipotiroidisme",
        "Hyperthyroidism": "Hipertiroidisme", "Hypoglycemia": "Hipoglikemia",
        "Osteoarthristis": "Osteoartritis", "Arthritis": "Artritis",
        "(vertigo) Paroymsal  Positional Vertigo": "Vertigo", "Acne": "Jerawat",
        "Urinary tract infection": "Infeksi Saluran Kemih",
        "Psoriasis": "Psoriasis", "Impetigo": "Impetigo"
    }

model, le, explainer = load_model()
symptoms, translation, doctor_map, prevention_map, df_desc, df_prec, df_encoded, severity_dict = load_data()
disease_translation = load_disease_translation()

def translate_symptom(s):
    return translation.get(s, s.replace('_', ' ').title())

def translate_disease(d):
    return disease_translation.get(d, d)

symptom_id_to_en = {translate_symptom(s): s for s in symptoms}
symptoms_indo    = sorted(symptom_id_to_en.keys())

def calc_severity(selected_en):
    total = sum(severity_dict.get(s, 0) for s in selected_en)
    max_possible = len(selected_en) * 7
    pct = (total / max_possible * 100) if max_possible > 0 else 0
    if pct < 35:
        return "Ringan", pct, "#22c55e", "🟢"
    elif pct < 65:
        return "Sedang", pct, "#f59e0b", "🟡"
    else:
        return "Berat", pct, "#ef4444", "🔴"

# Session state untuk riwayat
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 28px 0;'>
        <div style='font-size:3rem;'>🏥</div>
        <div style='font-family: Instrument Serif, serif; font-size:1.4rem; color:white; margin-top:8px;'>DeteksiPenyakit</div>
        <div style='font-size:0.78rem; opacity:0.6; margin-top:4px; color:white;'>Sistem Diagnosa Berbasis AI</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🔍  Prediksi Penyakit",
        "📚  Ensiklopedi Penyakit",
        "📜  Riwayat Pemeriksaan",
        "📊  Visualisasi Data",
        "ℹ️  Tentang Aplikasi"
    ])

    st.markdown(f"""
    <div style='margin-top:20px; padding:12px 14px; background:rgba(255,255,255,0.07); 
                border-radius:10px; font-size:0.8rem; color:rgba(255,255,255,0.7);'>
        📋 Riwayat: <b style='color:white;'>{len(st.session_state.history)} pemeriksaan</b>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HALAMAN 1 — PREDIKSI
# ════════════════════════════════════════════════════════════
if "Prediksi" in page:
    st.markdown("""
    <div class='hero'>
        <h1>🔍 Deteksi Penyakit dari Gejala</h1>
        <p>Pilih gejala yang kamu rasakan, sistem AI akan menganalisis dan memberikan prediksi penyakit beserta rekomendasi lengkap.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🩺 Pilih Gejala yang Dirasakan</div>", unsafe_allow_html=True)
    selected_indo = st.multiselect("Ketik atau pilih gejala:", options=symptoms_indo,
                                   placeholder="Contoh: Demam Tinggi, Sakit Kepala, Mual...")
    selected_en = [symptom_id_to_en[s] for s in selected_indo]

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        predict_btn = st.button("🔍 Analisis Sekarang", use_container_width=True)
    with col_info:
        if selected_indo:
            st.markdown(f"<div style='padding:12px; color:#475569; font-size:0.9rem;'>✅ {len(selected_indo)} gejala dipilih</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:12px; color:#94a3b8; font-size:0.9rem;'>ℹ️ Pilih minimal 2 gejala untuk memulai analisis</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        if len(selected_en) < 2:
            st.markdown("<div class='warning-box'>⚠️ Silakan pilih minimal <strong>2 gejala</strong> untuk mendapatkan hasil prediksi yang akurat.</div>", unsafe_allow_html=True)
        else:
            input_vec  = pd.DataFrame([{s: 1 if s in selected_en else 0 for s in symptoms}])
            proba      = model.predict_proba(input_vec)[0]
            top3_idx   = proba.argsort()[-3:][::-1]
            top3_en    = le.inverse_transform(top3_idx)
            top3_names = [translate_disease(d) for d in top3_en]
            top3_proba = proba[top3_idx]
            best_en    = top3_en[0]
            best_id    = top3_names[0]

            # Simpan ke riwayat
            st.session_state.history.append({
                "waktu"   : datetime.now().strftime("%d %b %Y, %H:%M"),
                "gejala"  : selected_indo,
                "penyakit": best_id,
                "persen"  : f"{top3_proba[0]*100:.1f}%"
            })

            # ── Hasil prediksi ──
            st.markdown("<div class='section-title' style='margin-top:8px;'>🎯 Hasil Prediksi</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            for col, cls, medal, name, pct in zip(
                [c1,c2,c3], ["gold","silver","bronze"],
                ["🥇","🥈","🥉"], top3_names, top3_proba
            ):
                with col:
                    st.markdown(f"""
                    <div class='result-card {cls}'>
                        <div class='result-medal'>{medal}</div>
                        <div class='result-name'>{name}</div>
                        <div class='result-pct'>{pct*100:.1f}%</div>
                        <div style='font-size:0.75rem;color:#94a3b8;margin-top:4px;'>Kemungkinan</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # ── Tingkat keparahan ──
            sev_label, sev_pct, sev_color, sev_icon = calc_severity(selected_en)
            st.markdown("<div class='section-title'>🌡️ Tingkat Keparahan Gejala</div>", unsafe_allow_html=True)
            col_sev1, col_sev2 = st.columns([2,1])
            with col_sev1:
                st.markdown(f"""
                <div class='card' style='margin-bottom:0;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                        <span style='font-weight:600; color:#1e293b;'>Skor Keparahan</span>
                        <span style='font-weight:800; color:{sev_color};'>{sev_icon} {sev_label}</span>
                    </div>
                    <div class='severity-bar-wrap'>
                        <div class='severity-bar' style='width:{sev_pct:.0f}%; background:{sev_color};'></div>
                    </div>
                    <div style='display:flex; justify-content:space-between; font-size:0.8rem; color:#94a3b8; margin-top:4px;'>
                        <span>Ringan</span><span>Sedang</span><span>Berat</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            with col_sev2:
                urgency = "Segera ke UGD" if sev_label == "Berat" else ("Kunjungi dokter" if sev_label == "Sedang" else "Bisa ke Puskesmas")
                urgency_color = "#ef4444" if sev_label == "Berat" else ("#f59e0b" if sev_label == "Sedang" else "#22c55e")
                st.markdown(f"""
                <div class='card' style='margin-bottom:0; text-align:center;'>
                    <div style='font-size:0.8rem; color:#64748b; margin-bottom:6px;'>Rekomendasi</div>
                    <div style='font-weight:700; color:{urgency_color}; font-size:0.95rem;'>{urgency}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # ── Dokter & Deskripsi ──
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(f"<div class='section-title'>📋 Tentang {best_id}</div>", unsafe_allow_html=True)
                if best_en in df_desc.index:
                    st.markdown(f"<div class='info-box'>{df_desc.loc[best_en, 'Description']}</div>", unsafe_allow_html=True)

                st.markdown("<div class='section-title' style='margin-top:16px;'>👨‍⚕️ Dokter yang Tepat</div>", unsafe_allow_html=True)
                dokter = doctor_map.get(best_en, "Dokter Umum")
                st.markdown(f"""
                <div class='doctor-box'>
                    <div style='font-size:2.2rem;'>🩺</div>
                    <div>
                        <div style='font-size:0.78rem; color:#64748b; margin-bottom:3px;'>Direkomendasikan ke</div>
                        <div style='font-weight:700; color:#1e3a5f; font-size:1rem;'>{dokter}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            with col_right:
                st.markdown("<div class='section-title'>⚠️ Tindakan Pertama</div>", unsafe_allow_html=True)
                if best_en in df_prec.index:
                    row = df_prec.loc[best_en]
                    icons = ["💊","🏃","🥗","🏥"]
                    for i in range(1, 5):
                        val = row.get(f'Precaution_{i}', '')
                        if pd.notna(val) and val:
                            st.markdown(f"""
                            <div class='precaution-item'>
                                <span style='font-size:1.2rem;'>{icons[i-1]}</span>
                                <span>{val.strip().capitalize()}</span>
                            </div>""", unsafe_allow_html=True)

                st.markdown("<div class='section-title' style='margin-top:16px;'>🛡️ Cara Pencegahan</div>", unsafe_allow_html=True)
                tips = prevention_map.get(best_en, [])
                for tip in tips:
                    st.markdown(f"""
                    <div class='prevention-item'>
                        <span style='font-size:1rem;'>✦</span>
                        <span>{tip}</span>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # ── SHAP ──
            st.markdown("<div class='section-title'>🧠 Analisis Pengaruh Gejala (SHAP)</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.85rem;color:#64748b;margin-bottom:16px;'>Seberapa besar kontribusi setiap gejala terhadap hasil prediksi.</div>", unsafe_allow_html=True)

            shap_vals     = explainer.shap_values(input_vec)
            best_idx_num  = top3_idx[0]
            shap_for_best = shap_vals[:, :, best_idx_num][0]

            shap_df = pd.DataFrame({
                'Gejala'    : [translate_symptom(s) for s in symptoms],
                'Kontribusi': shap_for_best
            })
            shap_df = shap_df[shap_df['Kontribusi'] != 0].sort_values('Kontribusi')

            fig = go.Figure(go.Bar(
                x=shap_df['Kontribusi'], y=shap_df['Gejala'],
                orientation='h', marker_line_width=0,
                marker_color=['#ef4444' if v > 0 else '#3b82f6' for v in shap_df['Kontribusi']]
            ))
            fig.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font_family='Plus Jakarta Sans',
                height=max(300, len(shap_df)*36),
                margin=dict(l=10, r=20, t=10, b=10),
                xaxis=dict(title='Nilai Kontribusi', showgrid=True, gridcolor='#f1f5f9',
                           zeroline=True, zerolinecolor='#cbd5e1'),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div style='background:#fef2f2;border:1px solid #fecaca;border-radius:12px;
                        padding:14px 18px;font-size:0.85rem;color:#991b1b;margin-top:8px;'>
                ⚠️ <strong>Disclaimer:</strong> Hasil prediksi ini hanya bersifat informatif dan tidak menggantikan
                diagnosis dokter. Segera konsultasikan dengan tenaga medis profesional.
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HALAMAN 2 — ENSIKLOPEDI
# ════════════════════════════════════════════════════════════
elif "Ensiklopedi" in page:
    st.markdown("""
    <div class='hero'>
        <h1>📚 Ensiklopedi Penyakit</h1>
        <p>Pelajari informasi lengkap tentang semua penyakit yang dapat dideteksi oleh sistem ini.</p>
    </div>
    """, unsafe_allow_html=True)

    search = st.text_input("🔍 Cari penyakit...", placeholder="Contoh: Demam Berdarah, Diabetes...")

    all_diseases_en = list(disease_translation.keys())
    all_diseases_id = [translate_disease(d) for d in all_diseases_en]

    if search:
        filtered = [(en, id_) for en, id_ in zip(all_diseases_en, all_diseases_id)
                    if search.lower() in id_.lower()]
    else:
        filtered = list(zip(all_diseases_en, all_diseases_id))

    for en, id_ in sorted(filtered, key=lambda x: x[1]):
        with st.expander(f"🦠 {id_}"):
            col1, col2 = st.columns(2)
            with col1:
                if en in df_desc.index:
                    st.markdown(f"**📋 Deskripsi**")
                    st.markdown(f"<div class='info-box'>{df_desc.loc[en, 'Description']}</div>", unsafe_allow_html=True)
                dokter = doctor_map.get(en, "Dokter Umum")
                st.markdown(f"""
                <div class='doctor-box' style='margin-top:12px;'>
                    <div style='font-size:1.8rem;'>🩺</div>
                    <div>
                        <div style='font-size:0.75rem;color:#64748b;'>Dokter yang Tepat</div>
                        <div style='font-weight:700;color:#1e3a5f;'>{dokter}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            with col2:
                if en in df_prec.index:
                    st.markdown("**⚠️ Tindakan Pertama**")
                    row = df_prec.loc[en]
                    icons = ["💊","🏃","🥗","🏥"]
                    for i in range(1,5):
                        val = row.get(f'Precaution_{i}','')
                        if pd.notna(val) and val:
                            st.markdown(f"<div class='precaution-item'><span>{icons[i-1]}</span><span>{val.strip().capitalize()}</span></div>", unsafe_allow_html=True)
                tips = prevention_map.get(en, [])
                if tips:
                    st.markdown("**🛡️ Cara Pencegahan**")
                    for tip in tips:
                        st.markdown(f"<div class='prevention-item'><span>✦</span><span>{tip}</span></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HALAMAN 3 — RIWAYAT
# ════════════════════════════════════════════════════════════
elif "Riwayat" in page:
    st.markdown("""
    <div class='hero'>
        <h1>📜 Riwayat Pemeriksaan</h1>
        <p>Lihat semua hasil analisis yang telah kamu lakukan selama sesi ini.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class='card' style='text-align:center; padding:48px;'>
            <div style='font-size:3rem; margin-bottom:12px;'>📋</div>
            <div style='font-weight:600; color:#1e293b; margin-bottom:8px;'>Belum ada riwayat pemeriksaan</div>
            <div style='color:#64748b; font-size:0.9rem;'>Lakukan prediksi terlebih dahulu di halaman Prediksi Penyakit</div>
        </div>""", unsafe_allow_html=True)
    else:
        col_info, col_clear = st.columns([3,1])
        with col_info:
            st.markdown(f"<div style='color:#64748b; font-size:0.9rem; padding:8px 0;'>Total {len(st.session_state.history)} pemeriksaan dalam sesi ini</div>", unsafe_allow_html=True)
        with col_clear:
            if st.button("🗑️ Hapus Semua"):
                st.session_state.history = []
                st.rerun()

        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"""
            <div class='history-item'>
                <div>
                    <div style='font-weight:700; color:#1e293b; margin-bottom:4px;'>
                        {item['penyakit']} <span class='tag'>{item['persen']}</span>
                    </div>
                    <div style='font-size:0.82rem; color:#64748b;'>
                        🕐 {item['waktu']} &nbsp;·&nbsp; 🩺 {len(item['gejala'])} gejala dipilih
                    </div>
                    <div style='font-size:0.82rem; color:#94a3b8; margin-top:4px;'>
                        {', '.join(item['gejala'][:4])}{'...' if len(item['gejala']) > 4 else ''}
                    </div>
                </div>
                <div style='font-size:1.5rem;'>🦠</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HALAMAN 4 — VISUALISASI
# ════════════════════════════════════════════════════════════
elif "Visualisasi" in page:
    st.markdown("""
    <div class='hero'>
        <h1>📊 Visualisasi Dataset</h1>
        <p>Eksplorasi data penyakit dan gejala yang digunakan untuk melatih model prediksi.</p>
    </div>
    """, unsafe_allow_html=True)

    symptom_cols = [c for c in df_encoded.columns if c != 'Disease']
    c1, c2, c3, c4 = st.columns(4)
    for col, num, label, icon in [
        (c1, str(df_encoded['Disease'].nunique()), "Total Penyakit", "🦠"),
        (c2, str(len(symptom_cols)), "Total Gejala", "🩺"),
        (c3, str(len(df_encoded)), "Total Data", "📁"),
        (c4, "100%", "Akurasi Model", "🎯"),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='font-size:1.8rem;margin-bottom:4px;'>{icon}</div>
                <div class='stat-number'>{num}</div>
                <div class='stat-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🦠 Distribusi Data per Penyakit</div>", unsafe_allow_html=True)
    disease_counts = df_encoded['Disease'].value_counts().reset_index()
    disease_counts.columns = ['Disease', 'Jumlah']
    disease_counts['Penyakit'] = disease_counts['Disease'].apply(translate_disease)
    fig1 = go.Figure(go.Bar(
        x=disease_counts['Jumlah'], y=disease_counts['Penyakit'],
        orientation='h', text=disease_counts['Jumlah'], textposition='outside',
        marker=dict(color=disease_counts['Jumlah'], colorscale=[[0,'#bfdbfe'],[1,'#1a5276']], showscale=False),
    ))
    fig1.update_layout(paper_bgcolor='white', plot_bgcolor='white', font_family='Plus Jakarta Sans',
                       height=850, margin=dict(l=10,r=40,t=10,b=10),
                       xaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Jumlah Data'),
                       yaxis=dict(showgrid=False, categoryorder='total ascending'))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🩺 Top 20 Gejala Paling Sering Muncul</div>", unsafe_allow_html=True)
    top_sym = df_encoded[symptom_cols].sum().sort_values(ascending=False).head(20)
    top_sym_indo = pd.Series(top_sym.values, index=[translate_symptom(s) for s in top_sym.index])
    fig2 = go.Figure(go.Bar(
        x=top_sym_indo.values, y=top_sym_indo.index,
        orientation='h', text=top_sym_indo.values, textposition='outside',
        marker=dict(color=top_sym_indo.values, colorscale=[[0,'#fed7aa'],[1,'#c2410c']], showscale=False),
    ))
    fig2.update_layout(paper_bgcolor='white', plot_bgcolor='white', font_family='Plus Jakarta Sans',
                       height=550, margin=dict(l=10,r=40,t=10,b=10),
                       xaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Frekuensi'),
                       yaxis=dict(showgrid=False, categoryorder='total ascending'))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔥 Heatmap Gejala per Penyakit</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.85rem;color:#64748b;margin-bottom:16px;'>Semakin gelap warnanya, semakin sering gejala tersebut muncul pada penyakit itu.</div>", unsafe_allow_html=True)
    top_syms_en   = df_encoded[symptom_cols].sum().sort_values(ascending=False).head(20).index.tolist()
    top_syms_indo = [translate_symptom(s) for s in top_syms_en]
    heatmap_data  = df_encoded.groupby('Disease')[top_syms_en].mean()
    heatmap_data.index   = [translate_disease(d) for d in heatmap_data.index]
    heatmap_data.columns = top_syms_indo
    fig3 = px.imshow(heatmap_data, color_continuous_scale='Blues', aspect='auto')
    fig3.update_layout(paper_bgcolor='white', plot_bgcolor='white', font_family='Plus Jakarta Sans',
                       height=750, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HALAMAN 5 — TENTANG
# ════════════════════════════════════════════════════════════
elif "Tentang" in page:
    st.markdown("""
    <div class='hero'>
        <h1>ℹ️ Tentang Aplikasi</h1>
        <p>Informasi lengkap mengenai sistem deteksi penyakit berbasis kecerdasan buatan ini.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='card'>
            <div class='section-title'>🤖 Tentang Model</div>
            <div style='color:#475569;font-size:0.95rem;line-height:1.8;'>
                Aplikasi ini menggunakan algoritma <strong>Random Forest</strong> yang dilatih 
                menggunakan dataset publik berisi <strong>41 jenis penyakit</strong> dan 
                <strong>131 gejala unik</strong>.<br><br>
                Model mencapai akurasi <strong>100%</strong> pada data uji dengan fitur 
                explainability menggunakan <strong>SHAP</strong>.
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <div class='section-title'>📊 Detail Teknis</div>
            <div style='color:#475569;font-size:0.95rem;line-height:1.8;'>
                <b>Algoritma:</b> Random Forest (100 trees)<br>
                <b>Fitur:</b> 131 gejala (one-hot encoded)<br>
                <b>Kelas:</b> 41 penyakit<br>
                <b>Data latih:</b> 3.936 sampel<br>
                <b>Data uji:</b> 984 sampel<br>
                <b>Explainability:</b> SHAP TreeExplainer<br>
                <b>Framework:</b> Streamlit + Scikit-learn
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='section-title'>🗺️ Fitur Aplikasi</div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
            <div style='background:#f0f7ff;border-radius:10px;padding:14px;'>
                <b>🔍 Prediksi Penyakit</b><br>
                <span style='font-size:0.85rem;color:#64748b;'>Input gejala → prediksi top 3 penyakit beserta probabilitasnya</span>
            </div>
            <div style='background:#f0fdf4;border-radius:10px;padding:14px;'>
                <b>🌡️ Tingkat Keparahan</b><br>
                <span style='font-size:0.85rem;color:#64748b;'>Analisis severity gejala: Ringan / Sedang / Berat</span>
            </div>
            <div style='background:#fdf4ff;border-radius:10px;padding:14px;'>
                <b>👨‍⚕️ Rekomendasi Dokter</b><br>
                <span style='font-size:0.85rem;color:#64748b;'>Saran spesialis dokter yang tepat sesuai penyakit</span>
            </div>
            <div style='background:#fffbeb;border-radius:10px;padding:14px;'>
                <b>🛡️ Cara Pencegahan</b><br>
                <span style='font-size:0.85rem;color:#64748b;'>Tips pencegahan spesifik per penyakit</span>
            </div>
            <div style='background:#fff1f2;border-radius:10px;padding:14px;'>
                <b>📚 Ensiklopedi Penyakit</b><br>
                <span style='font-size:0.85rem;color:#64748b;'>Database lengkap 41 penyakit dengan info detail</span>
            </div>
            <div style='background:#f0f4f8;border-radius:10px;padding:14px;'>
                <b>📜 Riwayat Pemeriksaan</b><br>
                <span style='font-size:0.85rem;color:#64748b;'>Histori semua hasil prediksi selama sesi berlangsung</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='section-title'>⚠️ Disclaimer Penting</div>
        <div style='color:#991b1b;font-size:0.95rem;line-height:1.8;background:#fef2f2;padding:16px;border-radius:10px;'>
            Aplikasi ini dibuat untuk tujuan <strong>edukasi dan penelitian</strong> semata. 
            Hasil prediksi <strong>tidak dapat dijadikan sebagai diagnosis medis resmi</strong>. 
            Selalu konsultasikan kondisi kesehatan Anda dengan dokter atau tenaga medis profesional.
        </div>
    </div>""", unsafe_allow_html=True)