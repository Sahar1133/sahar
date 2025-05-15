import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# ====================== STYLING & SETUP ======================
st.set_page_config(
    page_title="AI Career Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6e8efb, #4a6cf7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Radio buttons */
    .stRadio>div {
        flex-direction: column;
        gap: 8px;
    }
    .stRadio>div>label {
        background: #f1f3ff;
        padding: 12px;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stRadio>div>label:hover {
        background: #e0e5ff;
    }
    
    /* Headers */
    h1 {
        color: #2c3e50;
        border-bottom: 2px solid #4a6cf7;
        padding-bottom: 10px;
    }
    h2 {
        color: #3498db;
    }
    
    /* Expanders */
    .stExpander {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ====================== DATA LOADING & PREPROCESSING ======================
@st.cache_data
def load_data():
    # If CSV not found, use demo data
    try:
        data = pd.read_csv("new_updated_data.xlsx")
    except FileNotFoundError:
        st.warning("⚠️ Dataset not found. Using demo data.")
        data = pd.DataFrame({
            'Interest': ['Technology', 'Business', 'Arts', 'Engineering', 'Medical'],
            'Work_Style': ['Independent', 'Collaborative', 'Flexible', 'Independent', 'Collaborative'],
            'Strengths': ['Analytical', 'Creative', 'Strategic', 'Practical', 'Analytical'],
            'Communication_Skills': ['Medium', 'High', 'Low', 'Medium', 'High'],
            'Leadership_Skills': ['Medium', 'High', 'Low', 'Medium', 'High'],
            'Teamwork_Skills': ['High', 'Medium', 'Low', 'High', 'Medium'],
            'GPA': [3.5, 3.8, 3.2, 3.9, 3.6],
            'Years_of_Experience': [5, 10, 2, 8, 12],
            'Predicted_Career_Field': [
                'Software Developer', 
                'Marketing Manager', 
                'Graphic Designer',
                'Data Scientist',
                'Doctor'
            ]
        })
    
    # Clean data
    if 'GPA' in data.columns:
        data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
        data['GPA'].fillna(data['GPA'].median(), inplace=True)
    
    return data

data = load_data()

# ====================== MODEL TRAINING ======================
def preprocess_data(data):
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        if col != 'Predicted_Career_Field':
            data[col] = le.fit_transform(data[col].astype(str))
    
    data['Predicted_Career_Field'] = le.fit_transform(data['Predicted_Career_Field'])
    return data, le

processed_data, target_le = preprocess_data(data.copy())

def train_model(data):
    X = data.drop('Predicted_Career_Field', axis=1)
    y = data['Predicted_Career_Field']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

model, accuracy = train_model(processed_data)

# ====================== QUESTIONNAIRE ======================
questions = {
    "Teamwork_Skills": [
        {
            "question": "In group projects, you typically:",
            "options": [
                {"text": "Work separately on your part", "value": "Low"},
                {"text": "Coordinate some with teammates", "value": "Medium"},
                {"text": "Actively collaborate throughout", "value": "High"}
            ]
        },
        {
            "question": "When a teammate struggles, you:",
            "options": [
                {"text": "Focus on your own work", "value": "Low"},
                {"text": "Help if they ask directly", "value": "Medium"},
                {"text": "Proactively offer assistance", "value": "High"}
            ]
        }
    ],
    "Communication_Skills": [
        {
            "question": "How comfortable are you presenting ideas?",
            "options": [
                {"text": "Very uncomfortable", "value": "Low"},
                {"text": "Somewhat comfortable", "value": "Medium"},
                {"text": "Very comfortable", "value": "High"}
            ]
        }
    ]
}

direct_input_features = {
    "GPA": {"type": "number", "min": 0.0, "max": 4.0, "step": 0.1, "default": 3.0},
    "Years_of_Experience": {"type": "number", "min": 0, "max": 50, "step": 1, "default": 5}
}

# ====================== STREAMLIT APP ======================
def main():
    apply_custom_css()
    
    st.title("🧠 AI Career Prediction System")
    st.markdown("Discover your ideal career path based on your skills and preferences.")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("This tool uses machine learning to match your profile with suitable careers.")
    st.sidebar.write(f"Model Accuracy: **{accuracy:.1%}**")
    
    # Tabs
    tab1, tab2 = st.tabs(["Career Prediction", "Data Insights"])
    
    with tab1:
        st.header("📝 Career Assessment")
        user_responses = {}
        
        # Direct inputs (GPA, Experience)
        with st.expander("Academic & Professional Background"):
            for feature, config in direct_input_features.items():
                user_responses[feature] = st.number_input(
                    f"Your {feature.replace('_', ' ')}:",
                    min_value=config["min"],
                    max_value=config["max"],
                    value=config["default"],
                    step=config["step"]
                )
        
        # Questionnaire
        for feature, question_list in questions.items():
            with st.expander(f"🔹 {feature.replace('_', ' ')}"):
                for i, question in enumerate(question_list):
                    selected_option = st.radio(
                        question["question"],
                        [opt["text"] for opt in question["options"]],
                        key=f"{feature}_{i}"
                    )
                    selected_value = question["options"][[opt["text"] for opt in question["options"]].index(selected_option)]["value"]
                    if feature not in user_responses:
                        user_responses[feature] = []
                    user_responses[feature].append(selected_value)
        
        # Prediction
        if st.button("🚀 Predict My Career"):
            if len(user_responses) < 3:
                st.warning("Please answer more questions for better accuracy.")
            else:
                with st.spinner("Analyzing your profile..."):
                    # Prepare input data
                    input_data = processed_data.drop('Predicted_Career_Field', axis=1).iloc[0:1].copy()
                    for col in input_data.columns:
                        if col in user_responses:
                            if isinstance(user_responses[col], list):  # For question-based features
                                level_map = {"Low": 0, "Medium": 1, "High": 2}
                                avg_level = np.mean([level_map[val] for val in user_responses[col]])
                                input_data[col] = avg_level
                            else:  # For direct inputs
                                input_data[col] = user_responses[col]
                        else:
                            input_data[col] = processed_data[col].median()
                    
                    # Make prediction
                    prediction = model.predict(input_data)
                    predicted_career = target_le.inverse_transform(prediction)[0]
                    
                    # Explain prediction
                    st.success(f"🎯 **Recommended Career:** {predicted_career}")
                    
                    with st.expander("🔍 Why this recommendation?"):
                        st.write("The AI considered these key factors:")
                        
                        # Get feature importances
                        feat_importances = pd.Series(model.feature_importances_, index=input_data.columns)
                        top_features = feat_importances.sort_values(ascending=False).head(3)
                        
                        for feat in top_features.index:
                            st.write(f"- **{feat.replace('_', ' ')}** (importance: {top_features[feat]:.2f})")
                        
                        st.write("\n**Sample Decision Path:**")
                        st.code(export_text(model, feature_names=list(input_data.columns)).split('\n')[0])
    
    with tab2:
        st.header("📊 Dataset Insights")
        st.write("Explore the data used for predictions.")
        
        if st.checkbox("Show raw data"):
            st.dataframe(data)
        
        st.subheader("Career Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        data['Predicted_Career_Field'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Most Common Careers in Dataset")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
