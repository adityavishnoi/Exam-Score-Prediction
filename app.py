import streamlit as st
import pickle
import pandas as pd
import numpy as np


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ExamScore AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# LOAD MODEL AND ENCODERS
# =========================================================

@st.cache_resource
def load_model():

    # Load trained XGBoost model
    with open("best_xgb_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Load trained label encoders
    with open("label_encoders.pkl", "rb") as file:
        encoders = pickle.load(file)

    return model, encoders


try:
    model, encoders = load_model()

except Exception as e:

    st.error("Unable to load the model.")

    st.code(str(e))

    st.stop()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    /* -----------------------------------------------------
       GLOBAL
    ----------------------------------------------------- */

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {

        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99,102,241,0.18),
                transparent 25%
            ),

            radial-gradient(
                circle at 90% 20%,
                rgba(168,85,247,0.15),
                transparent 25%
            ),

            linear-gradient(
                135deg,
                #070b17 0%,
                #0d1224 50%,
                #080b16 100%
            );

        color: white;
    }


    /* -----------------------------------------------------
       HIDE STREAMLIT DEFAULT ELEMENTS
    ----------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* -----------------------------------------------------
       MAIN CONTAINER
    ----------------------------------------------------- */

    .block-container {

        padding-top: 2rem;
        padding-bottom: 3rem;

        max-width: 1250px;
    }


    /* -----------------------------------------------------
       HERO
    ----------------------------------------------------- */

    .hero {

        padding: 40px;

        border-radius: 28px;

        background:
            linear-gradient(
                135deg,
                rgba(99,102,241,0.20),
                rgba(168,85,247,0.12)
            );

        border: 1px solid rgba(255,255,255,0.10);

        box-shadow:
            0 25px 70px rgba(0,0,0,0.30);

        margin-bottom: 30px;
    }


    .hero-badge {

        display: inline-block;

        padding: 7px 15px;

        border-radius: 30px;

        background:
            rgba(99,102,241,0.18);

        border:
            1px solid rgba(129,140,248,0.30);

        color: #a5b4fc;

        font-size: 13px;

        font-weight: 600;

        margin-bottom: 15px;
    }


    .hero h1 {

        font-size: 48px;

        font-weight: 800;

        margin: 0;

        letter-spacing: -2px;

        background:
            linear-gradient(
                90deg,
                #ffffff,
                #c7d2fe,
                #d8b4fe
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;
    }


    .hero p {

        margin-top: 12px;

        color: #a5acc4;

        font-size: 17px;

        line-height: 1.6;
    }


    /* -----------------------------------------------------
       SECTION TITLE
    ----------------------------------------------------- */

    .section-title {

        font-size: 22px;

        font-weight: 700;

        margin:
            25px 0 15px 3px;
    }


    /* -----------------------------------------------------
       CARD
    ----------------------------------------------------- */

    .card {

        padding: 25px;

        border-radius: 20px;

        background:
            rgba(255,255,255,0.045);

        border:
            1px solid rgba(255,255,255,0.08);

        box-shadow:
            0 15px 40px rgba(0,0,0,0.18);
    }


    /* -----------------------------------------------------
       PREDICTION CARD
    ----------------------------------------------------- */

    .prediction-card {

        padding: 35px;

        border-radius: 25px;

        text-align: center;

        background:
            linear-gradient(
                145deg,
                rgba(99,102,241,0.20),
                rgba(168,85,247,0.10)
            );

        border:
            1px solid rgba(129,140,248,0.25);

        box-shadow:
            0 20px 60px rgba(79,70,229,0.15);
    }


    .prediction-label {

        color: #a5acc4;

        font-size: 14px;

        text-transform: uppercase;

        letter-spacing: 2px;
    }


    .prediction-score {

        font-size: 72px;

        font-weight: 800;

        margin: 8px 0;

        background:
            linear-gradient(
                90deg,
                #818cf8,
                #c084fc
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;
    }


    .prediction-sub {

        color: #c4c9dc;

        font-size: 15px;
    }


    /* -----------------------------------------------------
       EMPTY PREDICTION
    ----------------------------------------------------- */

    .empty-prediction {

        padding: 55px 30px;

        border-radius: 25px;

        text-align: center;

        background:
            rgba(255,255,255,0.035);

        border:
            1px dashed rgba(255,255,255,0.12);
    }


    .empty-icon {

        font-size: 45px;

        margin-bottom: 10px;
    }


    .empty-title {

        font-size: 20px;

        font-weight: 700;

        color: #e5e7eb;
    }


    .empty-text {

        color: #858ca5;

        margin-top: 8px;

        font-size: 14px;
    }


    /* -----------------------------------------------------
       METRIC CARDS
    ----------------------------------------------------- */

    .metric-card {

        padding: 20px;

        border-radius: 18px;

        background:
            rgba(255,255,255,0.035);

        border:
            1px solid rgba(255,255,255,0.07);
    }


    .metric-title {

        color: #858ca5;

        font-size: 13px;
    }


    .metric-value {

        color: white;

        font-size: 23px;

        font-weight: 700;

        margin-top: 5px;
    }


    /* -----------------------------------------------------
       SELECTBOX
    ----------------------------------------------------- */

    div[data-baseweb="select"] > div {

        background:
            rgba(255,255,255,0.045);

        border:
            1px solid rgba(255,255,255,0.12);

        border-radius: 12px;
    }


    /* -----------------------------------------------------
       INPUT LABELS
    ----------------------------------------------------- */

    label {

        color: #d7d9e5 !important;

        font-weight: 500 !important;
    }


    /* -----------------------------------------------------
       BUTTON
    ----------------------------------------------------- */

    .stButton > button {

        width: 100%;

        border: none;

        border-radius: 14px;

        padding: 15px 20px;

        font-size: 16px;

        font-weight: 700;

        color: white;

        background:
            linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6
            );

        box-shadow:
            0 10px 30px rgba(99,102,241,0.30);

        transition:
            all 0.25s ease;
    }


    .stButton > button:hover {

        transform:
            translateY(-2px);

        box-shadow:
            0 15px 40px rgba(139,92,246,0.40);
    }


    /* -----------------------------------------------------
       FOOTER
    ----------------------------------------------------- */

    .footer {

        text-align: center;

        color: #686f89;

        font-size: 13px;

        margin-top: 50px;

        line-height: 1.8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">
            ✦ MACHINE LEARNING • XGBOOST
        </div>
        <h1>
            ExamScore AI
        </h1>
        <p>
            Predict your expected exam score using study habits,
            attendance, sleep patterns and learning environment.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MAIN LAYOUT
# =========================================================

left, right = st.columns(
    [1.35, 0.85],
    gap="large"
)


# =========================================================
# INPUT SECTION
# =========================================================

with left:

    st.markdown(
        '<div class="section-title">📊 Student Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # LEFT INPUTS
    # -----------------------------------------------------

    with col1:

        study_hours = st.slider(
            "📚 Study Hours per Day",

            min_value=0.0,

            max_value=15.0,

            value=5.0,

            step=0.5
        )


        class_attendance = st.slider(
            "🏫 Class Attendance (%)",

            min_value=0,

            max_value=100,

            value=75,

            step=1
        )


        sleep_hours = st.slider(
            "😴 Sleep Hours per Day",

            min_value=0.0,

            max_value=12.0,

            value=7.0,

            step=0.5
        )


    # -----------------------------------------------------
    # RIGHT INPUTS
    # -----------------------------------------------------

    with col2:

        # Get actual encoder classes
        # so the UI matches your trained model.

        sleep_options = list(
            encoders["sleep_quality"].classes_
        )

        study_options = list(
            encoders["study_method"].classes_
        )

        facility_options = list(
            encoders["facility_rating"].classes_
        )


        sleep_quality = st.selectbox(
            "🌙 Sleep Quality",

            options=sleep_options
        )


        study_method = st.selectbox(
            "🧠 Study Method",

            options=study_options
        )


        facility_rating = st.selectbox(
            "🏛️ Facility Rating",

            options=facility_options
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    st.markdown("<br>", unsafe_allow_html=True)


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    predict = st.button(
        "🚀 Predict My Exam Score"
    )


# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

try:

    sleep_quality_encoded = (
        encoders["sleep_quality"]
        .transform([sleep_quality])[0]
    )


    study_method_encoded = (
        encoders["study_method"]
        .transform([study_method])[0]
    )


    facility_rating_encoded = (
        encoders["facility_rating"]
        .transform([facility_rating])[0]
    )

except Exception as e:

    st.error(
        f"Encoding error: {e}"
    )

    st.stop()


# =========================================================
# CREATE MODEL INPUT
# =========================================================

input_data = pd.DataFrame({

    "study_hours": [
        study_hours
    ],

    "class_attendance": [
        class_attendance
    ],

    "sleep_hours": [
        sleep_hours
    ],

    "sleep_quality": [
        sleep_quality_encoded
    ],

    "study_method": [
        study_method_encoded
    ],

    "facility_rating": [
        facility_rating_encoded
    ]
})


# =========================================================
# MAKE PREDICTION ONLY WHEN BUTTON IS CLICKED
# =========================================================

if predict:

    try:

        result = model.predict(
            input_data
        )

        st.session_state.prediction = float(
            result[0]
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        st.session_state.prediction = None


# =========================================================
# PREDICTION DISPLAY
# =========================================================

with right:

    st.markdown(
        '<div class="section-title">🎯 Prediction</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # BEFORE PREDICTION
    # -----------------------------------------------------

    if st.session_state.prediction is None:

        st.markdown(
            """
            <div class="empty-prediction">
                <div class="empty-icon">
                    🎓
                </div>
                <div class="empty-title">
                    Ready to Predict
                </div>
                <div class="empty-text">
                    Adjust the student profile and click
                    <b>Predict My Exam Score</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # AFTER PREDICTION
    # -----------------------------------------------------

    else:

        prediction = (
            st.session_state.prediction
        )

        # Keep displayed score in 0–100
        display_score = np.clip(
            prediction,
            0,
            100
        )


        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Predicted Exam Score
                </div>
                <div class="prediction-score">
                    {display_score:.1f}
                </div>
                <div class="prediction-sub">
                    Estimated score out of 100
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # PERFORMANCE INTERPRETATION
        # -------------------------------------------------

        if display_score >= 90:

            status = "Exceptional Performance 🚀"

        elif display_score >= 75:

            status = "Strong Performance 🔥"

        elif display_score >= 60:

            status = "Good Potential 👍"

        elif display_score >= 40:

            status = "Room for Improvement 📈"

        else:

            status = "Needs Stronger Preparation 💪"


        st.markdown(
            f"""
            <div class="card"
                 style="text-align:center;">
                <div style="
                    color:#858ca5;
                    font-size:13px;
                    margin-bottom:8px;
                ">
                    PERFORMANCE INDICATOR
                </div>
                <div style="
                    font-size:19px;
                    font-weight:700;
                    color:#e5e7eb;
                ">
                    {status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# PROFILE SUMMARY
# =========================================================

st.markdown(
    '<div class="section-title">📌 Your Prediction Profile</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Study Time
            </div>
            <div class="metric-value">
                {study_hours} hrs
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Attendance
            </div>
            <div class="metric-value">
                {class_attendance}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Sleep
            </div>
            <div class="metric-value">
                {sleep_hours} hrs
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Study Method
            </div>
            <div class="metric-value">
                {study_method.title()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# STUDENT METRICS
# =========================================================

st.markdown(
    '<div class="section-title">📈 Student Metrics</div>',
    unsafe_allow_html=True
)


chart_data = pd.DataFrame({

    "Metric": [
        "Study Hours",
        "Attendance",
        "Sleep Hours"
    ],

    "Value": [

        (study_hours / 15) * 100,

        class_attendance,

        (sleep_hours / 12) * 100

    ]
})


st.bar_chart(
    chart_data.set_index("Metric"),

    height=250
)


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander(
    "🤖 About this AI Model"
):
    st.markdown(
        """
        ### ExamScore AI

        This application uses an **XGBoost Regression**
        model to predict a student's expected exam score.

        **Model Features**

        • 📚 Study Hours  
        • 🏫 Class Attendance  
        • 😴 Sleep Hours  
        • 🌙 Sleep Quality  
        • 🧠 Study Method  
        • 🏛️ Facility Rating  

        The categorical variables are transformed using
        the **same LabelEncoders used during model training**.

        The prediction is generated only after clicking
        the **Predict My Exam Score** button.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        ExamScore AI • Machine Learning Prediction System
        <br>
        Built with Python • XGBoost • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)