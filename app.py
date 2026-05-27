import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from groq import Groq
import os
from dotenv import load_dotenv
import sqlite3
import bcrypt
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io

load_dotenv()

# Load models
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Patient Readmission Risk Dashboard",
    page_icon="+",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

    * { font-family: 'Syne', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #060612 0%, #0a0d1a 50%, #060e1f 100%);
        color: #e2e8f0;
    }

    .main-header {
        background: linear-gradient(135deg, #0f2744 0%, #091a2f 100%);
        border: 1px solid rgba(99,179,237,0.3);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 0 60px rgba(45,106,159,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }

    .risk-high {
        background: linear-gradient(135deg, #2d0000 0%, #1a0000 100%);
        border: 1px solid rgba(255,68,68,0.6);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 0 40px rgba(255,68,68,0.15);
        text-align: center;
    }

    .risk-medium {
        background: linear-gradient(135deg, #2d1500 0%, #1a0d00 100%);
        border: 1px solid rgba(255,153,68,0.6);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 0 40px rgba(255,153,68,0.15);
        text-align: center;
    }

    .risk-low {
        background: linear-gradient(135deg, #002d00 0%, #001a00 100%);
        border: 1px solid rgba(68,255,136,0.6);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 0 40px rgba(68,255,136,0.15);
        text-align: center;
    }

    .stat-card {
        background: linear-gradient(135deg, #0d1829 0%, #0a1220 100%);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.4rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2d6a9f 0%, #1a4a7a 100%);
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        font-family: 'Syne', sans-serif !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #3d8ac0 0%, #2a6a9a 100%) !important;
        box-shadow: 0 0 25px rgba(45,106,159,0.5) !important;
        transform: translateY(-2px) !important;
    }

    .stTextInput > div > div > input {
        background: #0d1829 !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        padding: 0.75rem !important;
        font-family: 'Syne', sans-serif !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #0d1829;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(99,179,237,0.2);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #718096;
        font-weight: 600;
        font-family: 'Syne', sans-serif;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2d6a9f, #1a4a7a) !important;
        color: white !important;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060612 0%, #0a0d1a 100%);
        border-right: 1px solid rgba(99,179,237,0.2);
    }

    .care-plan-box {
        background: linear-gradient(135deg, #0d1829 0%, #091422 100%);
        border: 1px solid rgba(99,179,237,0.25);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Database functions

def get_db():
    return sqlite3.connect('users.db')

def register_user(username, password):
    try:
        conn = get_db()
        c = conn.cursor()
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, hashed))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def login_user(username, password):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row and bcrypt.checkpw(password.encode(), row[0]):
        return True
    return False

def save_prediction(username, risk_score, risk_tier, age, time_in_hospital,
                    num_medications, number_inpatient, number_diagnoses):
    conn = get_db()
    c = conn.cursor()
    c.execute("""INSERT INTO predictions
                 (username, risk_score, risk_tier, age, time_in_hospital,
                  num_medications, number_inpatient, number_diagnoses)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (username, risk_score, risk_tier, age, time_in_hospital,
               num_medications, number_inpatient, number_diagnoses))
    conn.commit()
    conn.close()

def get_predictions(username):
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE username=? ORDER BY created_at DESC",
        conn, params=(username,))
    conn.close()
    return df

# Session state

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

# Login / Register Page

if not st.session_state.logged_in:
    col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
    with col_c2:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0 2rem 0;">
            <h1 style="font-size:2.5rem; font-weight:800; color:#63b3ed; margin:0.5rem 0;">
                Readmission Risk AI
            </h1>
            <p style="color:#718096; font-size:1rem; margin-bottom:2rem;">
                Clinical decision support powered by machine learning
            </p>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            if st.button("Login", key="login_btn"):
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            new_username = st.text_input("Choose Username", key="reg_user", placeholder="Pick a username")
            new_password = st.text_input("Choose Password", type="password", key="reg_pass", placeholder="Pick a strong password")
            if st.button("Create Account", key="reg_btn"):
                if register_user(new_username, new_password):
                    st.success("Account created! Please login.")
                else:
                    st.error("Username already exists")

# Main Dashboard

else:
    st.markdown(f"""
    <div class="main-header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h1 style="margin:0; font-size:2rem; font-weight:800; color:#63b3ed;">
                    Patient Readmission Risk Dashboard
                </h1>
                <p style="margin:0.5rem 0 0 0; color:#718096; font-size:0.95rem;">
                    AI-powered 30-day readmission prediction Â· 101,766 patient dataset
                </p>
            </div>
            <div style="text-align:right;">
                <p style="margin:0; color:#a0aec0; font-size:0.85rem;">Logged in as</p>
                <p style="margin:0; color:#63b3ed; font-weight:700; font-size:1.1rem;">{st.session_state.username}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.rerun()

    st.markdown("---")

    # Sidebar
    st.sidebar.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h3 style="color:#63b3ed; margin:0.5rem 0; font-weight:700;">Patient Details</h3>
        <p style="color:#718096; font-size:0.8rem;">Adjust values and click Predict</p>
    </div>
    """, unsafe_allow_html=True)

    age_input = st.sidebar.slider("Patient Age", 1, 100, 55)
    if age_input < 10: age = 0
    elif age_input < 20: age = 1
    elif age_input < 30: age = 2
    elif age_input < 40: age = 3
    elif age_input < 50: age = 4
    elif age_input < 60: age = 5
    elif age_input < 70: age = 6
    elif age_input < 80: age = 7
    elif age_input < 90: age = 8
    else: age = 9

    time_in_hospital   = st.sidebar.slider("Days in Hospital", 1, 14, 3)
    num_medications    = st.sidebar.slider("Number of Medications", 1, 81, 15)
    number_inpatient   = st.sidebar.slider("Prior Inpatient Visits", 0, 21, 1)
    number_emergency   = st.sidebar.slider("Prior Emergency Visits", 0, 76, 0)
    number_outpatient  = st.sidebar.slider("Prior Outpatient Visits", 0, 42, 0)
    num_lab_procedures = st.sidebar.slider("Lab Procedures", 1, 132, 40)
    num_procedures     = st.sidebar.slider("Number of Procedures", 0, 6, 1)
    number_diagnoses   = st.sidebar.slider("Number of Diagnoses", 1, 16, 5)

    st.sidebar.markdown("---")
    predict_clicked = st.sidebar.button("Predict Readmission Risk")

    if predict_clicked:
        input_data = np.zeros(42)
        input_data[2]  = age
        input_data[6]  = time_in_hospital
        input_data[7]  = num_lab_procedures
        input_data[8]  = num_procedures
        input_data[9]  = num_medications
        input_data[10] = number_outpatient
        input_data[11] = number_emergency
        input_data[12] = number_inpatient
        input_data[16] = number_diagnoses

        df_input = pd.read_csv('cleaned_patient_data.csv')
        X = df_input.drop(columns=['readmitted'])
        input_df = pd.DataFrame([input_data], columns=X.columns)

        risk_proba   = model.predict_proba(input_df)[0][1]
        risk_percent = round(float(risk_proba) * 100, 1)

        if risk_percent >= 50:
            tier = "HIGH RISK"
            card_class = "risk-high"
            risk_color = "#ff4444"
        elif risk_percent >= 25:
            tier = "MEDIUM RISK"
            card_class = "risk-medium"
            risk_color = "#ff9944"
        else:
            tier = "LOW RISK"
            card_class = "risk-low"
            risk_color = "#44ff88"

        save_prediction(st.session_state.username, risk_percent, tier,
                        age_input, time_in_hospital, num_medications,
                        number_inpatient, number_diagnoses)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size:3.5rem; font-weight:800; line-height:1; color:{risk_color};">
                    {risk_percent}%
                </div>
                <div style="font-size:1.2rem; font-weight:700; margin:0.5rem 0;
                            letter-spacing:2px; color:{risk_color};">
                    {tier}
                </div>
                <div style="color:#718096; font-size:0.85rem;">
                    30-day readmission probability
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(risk_proba))

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Patient Summary**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Age", f"{age_input} yrs")
                st.metric("Days in Hospital", time_in_hospital)
                st.metric("Medications", num_medications)
            with c2:
                st.metric("Prior Admissions", number_inpatient)
                st.metric("Diagnoses", number_diagnoses)
                st.metric("Lab Procedures", num_lab_procedures)

        with col2:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            st.markdown("**Top Risk Factors (SHAP)**")
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('#0d1829')
            ax.set_facecolor('#0d1829')
            shap.summary_plot(shap_values, input_df, plot_type="bar",
                              max_display=5, show=False)
            ax.tick_params(colors='#a0aec0')
            ax.xaxis.label.set_color('#a0aec0')
            st.pyplot(fig)
            plt.close()

        # AI Care Plan
        st.markdown("---")
        st.markdown("### AI Generated Care Plan")
        with st.spinner("Generating personalized care plan..."):
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            prompt = f"""You are a clinical decision support assistant.

A diabetic patient has been assessed:
- Age: {age_input} years old
- Risk Score: {risk_percent}% â€” {tier}
- Days in Hospital: {time_in_hospital}
- Number of Medications: {num_medications}
- Prior Inpatient Visits: {number_inpatient}
- Prior Emergency Visits: {number_emergency}
- Number of Diagnoses: {number_diagnoses}
- Lab Procedures: {num_lab_procedures}

Generate a concise, structured care plan with:
1. Immediate Actions
2. Follow-up Recommendations
3. Lifestyle Advice
4. Warning Signs to Watch For

Keep it professional and actionable."""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )

            st.markdown(f"""
            <div class="care-plan-box">
                {response.choices[0].message.content.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            # PDF Download
        st.markdown("---")
        st.markdown("### Download Patient Report")

        def generate_pdf(patient_age, days, meds, admissions, emergency,
                         diagnoses, labs, risk_score, risk_tier, care_plan_text):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter,
                                    rightMargin=inch, leftMargin=inch,
                                    topMargin=inch, bottomMargin=inch)

            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle('title', fontSize=22, fontName='Helvetica-Bold',
                                          textColor=colors.HexColor('#1a4a7a'),
                                          alignment=TA_CENTER, spaceAfter=6)
            story.append(Paragraph("Patient Readmission Risk Report", title_style))

            sub_style = ParagraphStyle('sub', fontSize=10, fontName='Helvetica',
                                        textColor=colors.grey, alignment=TA_CENTER, spaceAfter=20)
            story.append(Paragraph("AI-powered 30-day readmission prediction", sub_style))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2d6a9f')))
            story.append(Spacer(1, 0.2*inch))

            # Risk Score
            if 'HIGH' in risk_tier:
                risk_color = colors.HexColor('#ff4444')
            elif 'MEDIUM' in risk_tier:
                risk_color = colors.HexColor('#ff9944')
            else:
                risk_color = colors.HexColor('#44cc88')

            risk_style = ParagraphStyle('risk', fontSize=32, fontName='Helvetica-Bold',
                                         textColor=risk_color, alignment=TA_CENTER, spaceAfter=4)
            story.append(Paragraph(f"{risk_score}%", risk_style))

            tier_style = ParagraphStyle('tier', fontSize=14, fontName='Helvetica-Bold',
                                         textColor=risk_color, alignment=TA_CENTER, spaceAfter=20)
            story.append(Paragraph(risk_tier, tier_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
            story.append(Spacer(1, 0.2*inch))

            # Patient Info Table
            section_style = ParagraphStyle('section', fontSize=13, fontName='Helvetica-Bold',
                                            textColor=colors.HexColor('#1a4a7a'), spaceAfter=10)
            story.append(Paragraph("Patient Information", section_style))

            data = [
                ['Field', 'Value'],
                ['Age', f'{patient_age} years'],
                ['Days in Hospital', str(days)],
                ['Number of Medications', str(meds)],
                ['Prior Inpatient Visits', str(admissions)],
                ['Prior Emergency Visits', str(emergency)],
                ['Number of Diagnoses', str(diagnoses)],
                ['Lab Procedures', str(labs)],
            ]

            table = Table(data, colWidths=[3*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a4a7a')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 11),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f0f4f8'), colors.white]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                ('PADDING', (0,0), (-1,-1), 8),
                ('ROUNDEDCORNERS', [4,4,4,4]),
            ]))
            story.append(table)
            story.append(Spacer(1, 0.3*inch))

            # Care Plan
            story.append(Paragraph("AI Generated Care Plan", section_style))
            body_style = ParagraphStyle('body', fontSize=10, fontName='Helvetica',
                                         textColor=colors.HexColor('#2d3748'),
                                         spaceAfter=6, leading=16)
            for line in care_plan_text.split('\n'):
                if line.strip():
                    story.append(Paragraph(line.strip(), body_style))

            story.append(Spacer(1, 0.3*inch))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

            # Footer
            footer_style = ParagraphStyle('footer', fontSize=8, fontName='Helvetica',
                                           textColor=colors.grey, alignment=TA_CENTER, spaceBefore=10)
            story.append(Paragraph(
                "This report is generated by an AI system and is intended for clinical decision support only. "
                "Always consult a qualified healthcare professional before making clinical decisions.",
                footer_style))

            doc.build(story)
            buffer.seek(0)
            return buffer

        # Generate and download
        care_plan_text = response.choices[0].message.content
        pdf_buffer = generate_pdf(
            age_input, time_in_hospital, num_medications,
            number_inpatient, number_emergency, number_diagnoses,
            num_lab_procedures, risk_percent, tier, care_plan_text
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name=f"patient_risk_report_{st.session_state.username}.pdf",
            mime="application/pdf"
        )

    # Prediction History
    st.markdown("---")
    st.markdown("### Your Prediction History")
    history = get_predictions(st.session_state.username)
    if len(history) == 0:
        st.info("No predictions yet â€” make your first prediction using the sidebar!")
    else:
        st.markdown(f"<p style='color:#718096;'>Total predictions: <b style='color:#63b3ed;'>{len(history)}</b></p>",
                    unsafe_allow_html=True)
        display_df = history[['created_at', 'risk_score', 'risk_tier', 'age',
                               'number_inpatient', 'num_medications', 'number_diagnoses']].copy()
        display_df.columns = ['Date', 'Risk Score (%)', 'Risk Tier', 'Age',
                               'Prior Admissions', 'Medications', 'Diagnoses']
        st.dataframe(display_df, use_container_width=True)

    # Model Comparison
    st.markdown("---")
    st.markdown("### Model Comparison")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class="stat-card">
            <p style="color:#718096; margin:0; font-size:0.8rem;">BEST MODEL</p>
            <p style="color:#63b3ed; margin:0; font-size:1.3rem; font-weight:700;">XGBoost</p>
            <p style="color:#e2e8f0; margin:0; font-size:2rem; font-weight:800;">68.56%</p>
            <p style="color:#718096; margin:0; font-size:0.75rem;">ROC-AUC Score</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="stat-card">
            <p style="color:#718096; margin:0; font-size:0.8rem;">SECOND</p>
            <p style="color:#63b3ed; margin:0; font-size:1.3rem; font-weight:700;">Random Forest</p>
            <p style="color:#e2e8f0; margin:0; font-size:2rem; font-weight:800;">66.15%</p>
            <p style="color:#718096; margin:0; font-size:0.75rem;">ROC-AUC Score</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="stat-card">
            <p style="color:#718096; margin:0; font-size:0.8rem;">THIRD</p>
            <p style="color:#63b3ed; margin:0; font-size:1.3rem; font-weight:700;">Logistic Regression</p>
            <p style="color:#e2e8f0; margin:0; font-size:2rem; font-weight:800;">64.62%</p>
            <p style="color:#718096; margin:0; font-size:0.75rem;">ROC-AUC Score</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    img = Image.open('roc_curves.png')
    st.image(img, caption='ROC Curve Comparison - All Models', use_column_width=True)
