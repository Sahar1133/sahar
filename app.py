import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import random

# ====================== STYLING & SETUP ======================
st.set_page_config(
    page_title="Career Match Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-size: 16px;
    }
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #667eea;
        padding-bottom: 12px;
    }
    .result-box {
        background: linear-gradient(135deg, #f5f7fa, #e4e8f0);
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ====================== DATA & MODEL ======================
@st.cache_data
def load_data():
    career_options = [
        'AI Engineer', 'Data Scientist', 'Cybersecurity Specialist',
        'UX Designer', 'Financial Analyst', 'Marketing Director',
        'Biomedical Researcher', 'Robotics Engineer', 'Game Developer',
        'Environmental Consultant', 'Corporate Lawyer', 'Clinical Psychologist'
    ]
    
    try:
        data = pd.read_excel("career_data.xlsx")
    except:
        data = pd.DataFrame({
            'Interest': np.random.choice(['Tech', 'Business', 'Science', 'Creative'], 200),
            'Skills': np.random.choice(['Analytical', 'Creative', 'Technical', 'Social'], 200),
            'Work_Preference': np.random.choice(['Team', 'Solo', 'Flexible'], 200),
            'Experience_Level': np.random.randint(0, 5, 200),
            'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 200),
            'Career': np.random.choice(career_options, 200)
        })
    return data

def train_model(data):
    processed_data = data.copy()
    le = LabelEncoder()
    
    for col in ['Interest', 'Skills', 'Work_Preference', 'Education', 'Career']:
        processed_data[col] = le.fit_transform(processed_data[col])
    
    X = processed_data.drop('Career', axis=1)
    y = processed_data['Career']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    model.fit(X_train, y_train)
    
    return model, le, accuracy_score(y_test, model.predict(X_test))

data = load_data()
model, label_encoder, accuracy = train_model(data)

# ====================== QUESTIONNAIRE ======================
def get_questions():
    return [
        {
            "text": "What type of projects excite you most?",
            "options": [
                "Building tech solutions",
                "Analyzing financial data",
                "Scientific research",
                "Creative design work"
            ],
            "feature": "Interest"
        },
        {
            "text": "Your strongest ability is:",
            "options": [
                "Problem-solving",
                "Innovation",
                "Technical skills",
                "People skills"
            ],
            "feature": "Skills"
        },
        {
            "text": "Preferred work style:",
            "options": [
                "Collaborative team",
                "Independent work",
                "Mix of both"
            ],
            "feature": "Work_Preference"
        }
    ]

# ====================== MAIN APP ======================
def main():
    apply_custom_css()
    st.title("💼 Career Match Pro")
    st.write("Discover your perfect career in just 3 questions!")
    
    # Questions
    responses = {}
    for i, q in enumerate(get_questions()):
        responses[q["feature"]] = st.radio(
            f"{i+1}. {q['text']}",
            q["options"],
            key=q["feature"]
        )
    
    # Additional info
    with st.expander("+ Additional Information"):
        responses["Experience_Level"] = st.slider(
            "Years of relevant experience:", 0, 20, 2)
        responses["Education"] = st.selectbox(
            "Highest education level:",
            ["High School", "Bachelor", "Master", "PhD"])
    
    # Prediction
    if st.button("🚀 Find My Career", type="primary"):
        with st.spinner("Analyzing your unique profile..."):
            # Prepare input
            input_data = pd.DataFrame([responses])
            for col in ['Interest', 'Skills', 'Work_Preference', 'Education']:
                input_data[col] = label_encoder.transform(input_data[col])
            
            # Predict
            prediction = model.predict(input_data)[0]
            career = label_encoder.inverse_transform([prediction])[0]
            
            # Display single result
            st.markdown(f"""
            <div class='result-box'>
                <h2 style='color:#667eea;'>Your Perfect Career Match:</h2>
                <h1 style='text-align:center; margin:20px 0;'>{career}</h1>
                <p>Based on your profile and our analysis of {len(data)} career paths</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simple explanation
            feat_importance = pd.Series(
                model.feature_importances_,
                index=input_data.columns
            ).sort_values(ascending=False)
            
            st.write("**Key factors in this match:**")
            for feat in feat_importance.index[:2]:
                st.write(f"- Your **{feat.replace('_', ' ').lower()}** ({responses[feat]})")

if __name__ == "__main__":
    main()
