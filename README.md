# 🎓 ExamScore AI — Exam Score Prediction

An **AI-powered exam score prediction system** built using **XGBoost Regression** and deployed with **Streamlit**.

ExamScore AI predicts a student's expected exam score based on study habits, attendance, sleep patterns, study method, and learning environment.

---

## 🚀 Live Demo

🔗 **Streamlit App:**
[Add your deployed Streamlit URL here]

---

## 📌 Project Overview

Student performance can be influenced by several academic and lifestyle factors.

This project uses a machine learning regression model to estimate an expected exam score from six important features:

* 📚 Study Hours
* 🏫 Class Attendance
* 😴 Sleep Hours
* 🌙 Sleep Quality
* 🧠 Study Method
* 🏛️ Facility Rating

The trained **XGBoost Regressor** processes these features and generates an estimated exam score.

---

## 🧠 Machine Learning Model

The project uses:

**XGBoost Regressor**

XGBoost was selected because it is a powerful gradient-boosting algorithm that can model complex relationships between input features and a continuous target variable.

### Target Variable

```text
exam_score
```

### Input Features

```text
study_hours
class_attendance
sleep_hours
sleep_quality
study_method
facility_rating
```

---

## 🔄 Machine Learning Pipeline

```text
Raw Student Data
       │
       ▼
Data Cleaning
       │
       ▼
Feature Selection
       │
       ▼
Categorical Encoding
       │
       ▼
Train / Test Split
       │
       ▼
XGBoost Regression
       │
       ▼
Model Evaluation
       │
       ▼
Save Model + Encoders
       │
       ▼
Streamlit Application
       │
       ▼
Predicted Exam Score
```

---

## 🎨 Streamlit Application

The application provides a modern interactive interface where users can enter their academic and lifestyle information.

### Features

* ✨ Modern dark-themed UI
* 📊 Interactive student profile
* 🎚️ Study hour slider
* 📈 Attendance slider
* 😴 Sleep hour input
* 🔽 Categorical feature selection
* 🚀 One-click prediction
* 🎯 Predicted exam score
* 📊 Student metrics visualization
* 🤖 Model information section
* 📱 Responsive Streamlit layout

---

## 🖥️ Application Workflow

1. Enter the student's study hours.
2. Set class attendance.
3. Enter average sleep hours.
4. Select sleep quality.
5. Select study method.
6. Select facility rating.
7. Click **Predict My Exam Score**.
8. The XGBoost model generates the predicted score.

---

## 📂 Project Structure

```text
ExamScore-AI/
│
├── app.py
├── best_xgb_model.pkl
├── label_encoders.pkl
├── requirements.txt
└── README.md
```

### File Description

| File                 | Description                          |
| -------------------- | ------------------------------------ |
| `app.py`             | Streamlit application                |
| `best_xgb_model.pkl` | Trained XGBoost regression model     |
| `label_encoders.pkl` | Trained categorical feature encoders |
| `requirements.txt`   | Python dependencies                  |
| `README.md`          | Project documentation                |

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* XGBoost
* Scikit-learn
* Pandas
* NumPy

### Visualization

* Streamlit
* Streamlit Charts

### Deployment

* Streamlit Community Cloud

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ExamScore-AI.git
```

Navigate to the project directory:

```bash
cd ExamScore-AI
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📦 Requirements

The project uses the following major dependencies:

```text
streamlit
pandas
numpy
scikit-learn
xgboost==3.1.3
```

---

## 🔐 Model & Encoder Handling

The project stores the trained model and categorical encoders separately:

```text
best_xgb_model.pkl
label_encoders.pkl
```

The same encoders used during model training are loaded during prediction to ensure that categorical values are transformed consistently.

---

## 📊 Prediction Features

| Feature            | Type        | Description                   |
| ------------------ | ----------- | ----------------------------- |
| `study_hours`      | Numerical   | Average study hours per day   |
| `class_attendance` | Numerical   | Attendance percentage         |
| `sleep_hours`      | Numerical   | Average sleep hours           |
| `sleep_quality`    | Categorical | Quality of sleep              |
| `study_method`     | Categorical | Preferred study method        |
| `facility_rating`  | Categorical | Rating of learning facilities |

---

## 🔮 Future Improvements

Possible improvements for future versions include:

* 📈 Model performance dashboard
* 📊 Feature importance visualization
* 🧠 Explainable AI using SHAP
* 📋 Prediction history
* 📥 Download prediction report
* 👤 Student profile management
* 📊 Multiple model comparison
* 🔄 Automated model retraining
* ☁️ Cloud-based model monitoring

---

## 🎯 Learning Outcomes

Through this project, I worked with:

* Regression machine learning
* XGBoost
* Feature engineering
* Categorical encoding
* Model serialization
* Streamlit application development
* Interactive UI design
* Machine learning deployment

---

## 👨‍💻 Author

**Aditya Vishnoi**

BCA Graduate | MCA Student | Aspiring AI/ML Engineer

🔗 Live Link:
https://exam-score-prediction-mhwt6uhqjs5lortzmckcbf.streamlit.app/

🔗 LinkedIn:
https://www.linkedin.com/in/aditya-vishnoi-7a92032b8

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

### 📄 License

This project is created for educational and portfolio purposes.
