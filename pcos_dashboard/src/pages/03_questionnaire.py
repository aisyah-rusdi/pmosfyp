import streamlit as st
import pandas as pd
import joblib
import os
from database.connection import get_connection
from utils.ui_helpers import load_css
from components.sidebar import render_sidebar
from components.header import render_header
from components.disclaimer import show_disclaimer_banner
from utils.questions import QUESTIONNAIRE

from database.queries import (
    GET_ACTIVE_QUESTIONNAIRE, 
    INSERT_SUBMISSION, 
    INSERT_ANSWER, 
    GET_ACTIVE_ML_MODEL, 
    INSERT_PREDICTION,
)

# ==========================================
# 1. PAGE CONFIGURATION & CACHING
# ==========================================
st.set_page_config(
    page_title="PCOS App - Questionnaire",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css("assets/style.css")
render_sidebar(current_page="questionnaire")
render_header(current_page="questionnaire")  
show_disclaimer_banner("This platform is not a diagnostic tool and does not replace professional clinical assessment.")

@st.cache_resource 
def load_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pages/ -> src/
    model = joblib.load(os.path.join(base_dir, 'models', 'pcos_stacking_model.joblib'))
    common_features = joblib.load(os.path.join(base_dir, 'models', 'common_features.joblib'))
    return model, common_features

model, common_features = load_models()

mappings = {
    "Blood Group": {
        "A+": 11, "A-": 12, "B+": 13, "B-": 14, 
        "O+": 15, "O-": 16, "AB+": 17, "AB-": 18
    },
    "YN": {"Yes": 1, "No": 0},
    "Cycle": {"Regular": 2, "Irregular": 4} 
}

# ==========================================
# 2. WIZARD SESSION STATE SETUP
# ==========================================
if "step" not in st.session_state:
    st.session_state.step = 1
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "error_msg" not in st.session_state:
    st.session_state.error_msg = ""
if "is_submitted" not in st.session_state:
    st.session_state.is_submitted = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# ==========================================
# 3. UI RENDERING HELPERS
# ==========================================
def render_input(q):
    """Renders individual input fields with dynamic placeholders."""
    widget_key = f"q_{q['id']}"
    existing_val = st.session_state.user_answers.get(q['id'], None)
    
    if q["type"] == "number":
        step_val = 1.0 if "years" in q['id'].lower() else 0.1
        min_v = float(q.get("min_value", 0.0))
        max_v = float(q.get("max_value")) if "max_value" in q else None
        
        st.number_input(
            q["id"], 
            min_value=min_v,
            max_value=max_v,
            value=float(existing_val) if existing_val is not None else None, 
            step=float(step_val), 
            key=widget_key, 
            placeholder="0.00"  
        )
    elif q["type"] == "text":
        st.text_input(
            q["id"], 
            value=str(existing_val) if existing_val is not None else "", 
            key=widget_key, 
            placeholder="Type here..."
        )
    elif q["type"] == "selectbox":
        opts = ["Select an option..."] + q.get("options", [])
        idx = opts.index(existing_val) if existing_val in opts else 0
        st.selectbox(q["id"], opts, index=idx, key=widget_key)
    elif q["type"] == "radio":
        opts = q.get("options", ["Yes", "No"]) 
        idx = opts.index(existing_val) if existing_val in opts else None
        st.radio(q["id"], options=opts, index=idx, horizontal=True, key=widget_key)

def render_question_block(q):
    """Renders a full row width OR splits into columns if there are follow-ups."""
    if "follow_up" in q:
        col_main, col_fu = st.columns([1, 4])
        with col_main:
            widget_key = f"q_{q['id']}"
            existing_val = st.session_state.user_answers.get(q['id'], None)
            opts = q.get("options", ["Yes", "No"])
            idx = opts.index(existing_val) if existing_val in opts else None
            current_selection = st.radio(q["id"], options=opts, index=idx, horizontal=True, key=widget_key)
            
        with col_fu:
            if current_selection == q["follow_up"]["condition"]:
                fus = q["follow_up"]["questions"]
                fu_cols = st.columns(len(fus), vertical_alignment="bottom") 
                for i, fu in enumerate(fus):
                    with fu_cols[i]:
                        render_input(fu)
    else:
        render_input(q)

# ==========================================
# 4. NAVIGATION & DATABASE LOGIC 
# ==========================================
def go_next(section_name):
    for q in QUESTIONNAIRE[section_name]:
        key = q['id']
        widget_key = f"q_{key}"
        if widget_key in st.session_state:
            st.session_state.user_answers[key] = st.session_state[widget_key]
            
        if "follow_up" in q:
            for fu in q["follow_up"]["questions"]:
                fu_key = fu["id"]
                fu_widget_key = f"q_{fu_key}"
                if fu_widget_key in st.session_state:
                    st.session_state.user_answers[fu_key] = st.session_state[fu_widget_key]
            
    missing = [
        q['id'] for q in QUESTIONNAIRE[section_name] 
        if q.get('type') != 'calculated' and (
            st.session_state.user_answers.get(q['id']) is None 
            or st.session_state.user_answers.get(q['id']) == "Select an option..."
            or st.session_state.user_answers.get(q['id']) == ""
        )
    ]
    
    if missing:
        st.session_state.error_msg = "Please answer all main questions on this page before proceeding."
    else:
        st.session_state.error_msg = ""
        st.session_state.step += 1

def go_back():
    st.session_state.error_msg = ""
    st.session_state.step -= 1

# ==========================================
# SAVES TO MARIADB
# ==========================================
def save_assessment_to_db(user_answers, prediction, probability, user_id):
    """Connects to MariaDB and saves the submission, answers, and prediction."""
    try:
        conn = get_connection() 
        cursor = conn.cursor()

        # 2. Get the active questionnaire
        cursor.execute(GET_ACTIVE_QUESTIONNAIRE)
        questionnaire_row = cursor.fetchone()
        if not questionnaire_row:
            raise Exception("No active questionnaire found in the database!")
        questionnaire_id = questionnaire_row['questionnaire_id']

        # 3. Create a new submission record
        cursor.execute(INSERT_SUBMISSION, (user_id, questionnaire_id))
        submission_id = cursor.lastrowid

        # 4. Build the translation dictionary
        cursor.execute("SELECT question_id, question_text FROM questions WHERE questionnaire_id = %s", (questionnaire_id,))
        q_mapping = {row['question_text']: row['question_id'] for row in cursor.fetchall()}

        # 5. Loop through answers and save each one
        for q_text, answer_val in user_answers.items():
            q_id = q_mapping.get(q_text)
            
            if not q_id:
                print(f"Warning: Could not find database ID for question: '{q_text}'")
                continue 
                
            num_val = None
            txt_val = None
            selected_option_id = None
            
            if isinstance(answer_val, (int, float)) and not isinstance(answer_val, bool):
                num_val = answer_val
            else:
                txt_val = str(answer_val)
                cursor.execute(
                    "SELECT option_id FROM options WHERE question_id = %s AND option_text = %s", 
                    (q_id, txt_val)
                )
                opt_row = cursor.fetchone()
                if opt_row:
                    selected_option_id = opt_row['option_id']
                
            cursor.execute(INSERT_ANSWER, (submission_id, q_id, selected_option_id, txt_val, num_val))

        # 6. Save the prediction
        cursor.execute(GET_ACTIVE_ML_MODEL)
        model_row = cursor.fetchone()
        model_id = model_row['model_id'] if model_row else 1

        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        cursor.execute(INSERT_PREDICTION, (submission_id, user_id, model_id, float(probability), risk_level))

        conn.commit()
        print("Successfully saved submission, answers, and prediction to database!")

    except Exception as e:
        print(f"Database Error: {e}")
        st.error(f"Failed to save to database. Error: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            if hasattr(conn, 'open') and conn.open:
                conn.close()


def process_and_predict():
    """Handles mapping, model prediction, and calling the DB save function."""
    final_data = {}
    feature_mapping = {
        "How old are you?": "age (yrs)",
        "How many years have you been married?": "marraige status (yrs)", 
        "What is your blood group?": "blood group",
        "Enter your weight (kg)": "weight (kg)",
        "Enter your height (cm)": "height(cm)",
        "Calculate your BMI": "bmi",
        "Enter your hip circumference (inch)": "hip(inch)",
        "Enter your waist circumference (inch)": "waist(inch)",
        "Calculate your Waist-Hip Ratio": "waist:hip ratio",
        "Enter your pulse rate (bpm)": "pulse rate(bpm)",
        "Enter your respiratory rate (breaths/min)": "rr (breaths/min)",
        "Enter your systolic blood pressure (mmHg)": "bp _systolic (mmhg)",
        "Enter your diastolic blood pressure (mmHg)": "bp _diastolic (mmhg)",
        "Enter your cycle length (days)": "cycle length(days)",
        "Is your cycle regular or irregular?": "cycle(r/i)",
        "Do you notice any skin darkening? (armpits, thighs, neck, etc)": "skin darkening (y/n)",
        "Do you notice any abrupt weight gain?": "weight gain(y/n)",
        "Do you notice hair growth at unexpected places? (chin, upper lip, abdomen, etc)": "hair growth(y/n)",
        "Do you have frequent pimples/acne?": "pimples(y/n)",
        "Do you have significant hair loss/thinning?": "hair loss(y/n)",
        "How many abortions have you had?": "no. of aborptions", 
        "Are you currently pregnant?": "pregnant(y/n)",
        "Do you eat Fast Food often?": "fast food (y/n)",
        "Do you excersize regularly?": "reg.exercise(y/n)"
    }
    
    for section in QUESTIONNAIRE.values():
        for q in section:
            q_id = q['id']
            if q_id in feature_mapping:
                raw_val = st.session_state.user_answers.get(q_id)
                model_col_name = feature_mapping[q_id]
                
                if q_id == "What is your blood group?":
                    final_data[model_col_name] = mappings["Blood Group"].get(raw_val, 0)
                elif q_id == "Is your cycle regular or irregular?":
                    final_data[model_col_name] = mappings["Cycle"].get(raw_val, 0)
                elif raw_val in ["Yes", "No"]:
                    final_data[model_col_name] = mappings["YN"][raw_val]
                else:
                    final_data[model_col_name] = raw_val

    input_df = pd.DataFrame([final_data])
    input_df = input_df[common_features]
    
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    st.session_state.prediction_result = {
        "status": int(prediction[0]),
        "probability": probability
    }
    
    # FIX: pass the real session user_id instead of hardcoded 1
    save_assessment_to_db(
        st.session_state.user_answers,
        int(prediction[0]),
        probability,
        st.session_state.get('user_id')  
    )
    
    st.session_state.is_submitted = True
    st.session_state.error_msg = ""

def submit_assessment(section_name):
    go_next(section_name)
    if not st.session_state.error_msg:
        process_and_predict()

# ==========================================
# 5. MAIN PAGE CONTENT (THE UI)
# ==========================================
st.title("Questionnaire")
st.markdown("Please fill out the details below so we can personalize your experience.")

safe_progress = min(st.session_state.step / 3.0, 1.0)
st.progress(safe_progress)
st.write("")

with st.container(border=True):
    
    # --- STEP 1: General Info ---
    if st.session_state.step == 1:
        st.markdown("### Step 1 of 3: 📋 General Information")
        st.divider()
        
        bmi_group = ["Enter your weight (kg)", "Enter your height (cm)", "Calculate your BMI"]
        whr_group = ["Enter your hip circumference (inch)", "Enter your waist circumference (inch)", "Calculate your Waist-Hip Ratio"]
        skip_list = []
        
        for q in QUESTIONNAIRE["general_information"]:
            if q['id'] in skip_list:
                continue
                
            if q['id'] in bmi_group:
                cols = st.columns(3)
                with cols[0]:
                    render_input(next(i for i in QUESTIONNAIRE["general_information"] if i["id"] == "Enter your weight (kg)"))
                with cols[1]:
                    render_input(next(i for i in QUESTIONNAIRE["general_information"] if i["id"] == "Enter your height (cm)"))
                with cols[2]:
                    bq = next(i for i in QUESTIONNAIRE["general_information"] if i["id"] == "Calculate your BMI")
                    weight = st.session_state.get("q_Enter your weight (kg)", None) or 0.0
                    height = st.session_state.get("q_Enter your height (cm)", None) or 0.0
                    bmi = 0.0
                    if weight > 0 and height > 0:
                        bmi = weight / ((height / 100) ** 2)
                    st.info(f"**Calculated BMI:**\n\n {bmi:.2f}")
                    st.session_state.user_answers[bq['id']] = bmi 
                skip_list.extend(bmi_group)
                st.write("")
                
            elif q['id'] in whr_group:
                cols = st.columns(3)
                with cols[0]:
                    render_input(next(i for i in QUESTIONNAIRE["general_information"] if i["id"] == "Enter your hip circumference (inch)"))
                with cols[1]:
                    render_input(next(i for i in QUESTIONNAIRE["general_information"] if i["id"] == "Enter your waist circumference (inch)"))
                with cols[2]:
                    rq = next(i for i in QUESTIONNAIRE["general_information"] if i["id"] == "Calculate your Waist-Hip Ratio")
                    hip = st.session_state.get("q_Enter your hip circumference (inch)", None) or 0.0
                    waist = st.session_state.get("q_Enter your waist circumference (inch)", None) or 0.0
                    whr = 0.0
                    if hip > 0:
                        whr = waist / hip
                    st.info(f"**Calculated Waist-Hip Ratio:**\n\n {whr:.2f}")
                    st.session_state.user_answers[rq['id']] = whr
                skip_list.extend(whr_group)
                st.write("")
                
            else:
                render_question_block(q)
                st.write("")

        if st.session_state.error_msg:
            st.error(st.session_state.error_msg)
        st.button("Next", type="primary", on_click=go_next, args=("general_information",))
        st.divider()
        show_disclaimer_banner("Your responses are used to personalise your experience and may be visible to your assigned professional.", icon="🔒")

    # --- STEP 2: Physical Symptoms ---
    elif st.session_state.step == 2:
        st.markdown("### Step 2 of 3: 🤒 Physical Symptoms")
        st.divider()
        
        for q in QUESTIONNAIRE["physical_symptoms"]:
            render_question_block(q)
            st.write("")

        if st.session_state.error_msg:
            st.error(st.session_state.error_msg)
            
        btn_col1, btn_col2, empty = st.columns([1, 1, 6])
        with btn_col1:
            st.button("Back", on_click=go_back)
        with btn_col2:
            st.button("Next", type="primary", on_click=go_next, args=("physical_symptoms",))
        
        st.divider()
        show_disclaimer_banner("Your responses are used to personalise your experience and may be visible to your assigned professional.", icon="🔒")

    # --- STEP 3: Lifestyle & Submission ---
    elif st.session_state.step >= 3:
        if st.session_state.is_submitted:
            st.success("🎉 Assessment Complete!")
            
            res = st.session_state.prediction_result
            if res["status"] == 1:
                st.error(f"High Risk of PCOS Detected ({res['probability']*100:.2f}%)")
            else:
                st.success(f"Low Risk of PCOS Detected ({res['probability']*100:.2f}%)")
                 
            if st.button("Start New Assessment"):
                st.session_state.step = 1
                st.session_state.user_answers = {}
                st.session_state.is_submitted = False
                st.session_state.prediction_result = None
                st.rerun()
        else:
            st.markdown("### Step 3 of 3: 🏃‍♀️ Lifestyle and Medical History")
            st.divider()
            
            for q in QUESTIONNAIRE["lifestyle_medical_history"]:
                render_question_block(q)
                st.write("")

            if st.session_state.error_msg:
                st.error(st.session_state.error_msg)
                
            btn_col1, btn_col2, empty = st.columns([1, 1.5, 5])
            with btn_col1:
                st.button("Back", on_click=go_back)
            with btn_col2:
                st.button("Submit Assessment", type="primary", on_click=submit_assessment, args=("lifestyle_medical_history",))
            st.divider()
            show_disclaimer_banner("Your responses are used to personalise your experience and may be visible to your assigned professional.", icon="🔒")