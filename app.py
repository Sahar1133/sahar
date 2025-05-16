import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ====================== STREAMLIT CONFIG & STYLE ======================
st.set_page_config(
    page_title="Career Path Predictor",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stButton>button {
        background: linear-gradient(135deg, #6e8efb, #4a6cf7);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-size: 16px;
    }
    .stButton>button:hover {
        transform: scale(1.02); box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1 { color: #2c3e50; border-bottom: 2px solid #4a6cf7; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    career_options = [
        'Software Developer', 'Data Scientist', 'AI Engineer', 
        'Cybersecurity Specialist', 'Cloud Architect', 'Marketing Manager',
        'Financial Analyst', 'HR Manager', 'Entrepreneur', 'Investment Banker',
        'Graphic Designer', 'Video Editor', 'Music Producer', 'Creative Writer',
        'Art Director', 'Mechanical Engineer', 'Electrical Engineer', 
        'Civil Engineer', 'Robotics Engineer', 'Doctor', 'Nurse', 'Psychologist',
        'Physical Therapist', 'Medical Researcher', 'Biotechnologist',
        'Research Scientist', 'Environmental Scientist', 'Physicist', 'Teacher',
        'Professor', 'Educational Consultant', 'Curriculum Developer', 'Lawyer',
        'Judge', 'Legal Consultant', 'UX Designer', 'Product Manager',
        'Journalist', 'Public Relations Specialist', 'Architect', 'Urban Planner',
        'Chef', 'Event Planner', 'Fashion Designer'
    ]

    try:
        data = pd.read_excel("new_updated_data.xlsx")
        if len(data['Predicted_Career_Field'].unique()) < 20:
            data['Predicted_Career_Field'] = np.random.choice(career_options, size=len(data))
    except:
        st.warning("⚠️ Dataset not found. Using demo data.")
        data = pd.DataFrame({
            'Interest': np.random.choice(['Technology', 'Business', 'Arts', 'Engineering', 'Medical', 'Science', 'Education', 'Law'], 200),
            'Work_Style': np.random.choice(['Independent', 'Collaborative', 'Flexible'], 200),
            'Strengths': np.random.choice(['Analytical', 'Creative', 'Strategic', 'Practical'], 200),
            'Communication_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Leadership_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Teamwork_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'GPA': np.round(np.random.uniform(2.0, 4.0, 200), 1),
            'Years_of_Experience': np.random.randint(0, 20, 200),
            'Predicted_Career_Field': np.random.choice(career_options, 200)
        })

    data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
    data['GPA'].fillna(data['GPA'].median(), inplace=True)
    return data

data = load_data()

# ====================== PREPROCESSING ======================
def preprocess_data(data):
    le = LabelEncoder()
    object_cols = [col for col in data.select_dtypes(include=['object']).columns if col != 'Predicted_Career_Field']
    for col in object_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    if 'Predicted_Career_Field' in data.columns:
        data['Predicted_Career_Field'] = LabelEncoder().fit_transform(data['Predicted_Career_Field'])

    return data, le

processed_data, target_le = preprocess_data(data.copy())

# ====================== MODEL ======================
def train_model(data):
    if 'Predicted_Career_Field' not in data.columns:
        st.error("Target column missing.")
        return None, 0

    X = data.drop('Predicted_Career_Field', axis=1)
    y = data['Predicted_Career_Field']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

model, accuracy = train_model(processed_data)

# ====================== QUESTIONS ======================
def get_all_questions():
    return [
        {"feature": "Interest", "question": "What are your main interests?", "options": ["Technology", "Business", "Arts", "Engineering", "Medical", "Science", "Education", "Law"]},
        {"feature": "Work_Style", "question": "Which work style suits you best?", "options": ["Independent", "Collaborative", "Flexible"]},
        {"feature": "Strengths", "question": "What best describes your strength?", "options": ["Analytical", "Creative", "Strategic", "Practical"]},
        {"feature": "Communication_Skills", "question": "How would you rate your communication skills?", "options": ["Low", "Medium", "High"]},
        {"feature": "Leadership_Skills", "question": "Rate your leadership skills.", "options": ["Low", "Medium", "High"]},
        {"feature": "Teamwork_Skills", "question": "How strong are your teamwork skills?", "options": ["Low", "Medium", "High"]}
    ]

def get_randomized_questions():
    all_qs = get_all_questions()
    features = list(set(q['feature'] for q in all_qs))
    selected = []

    for feature in features:
        q_pool = [q for q in all_qs if q['feature'] == feature]
        if q_pool:
            selected.append(random.choice(q_pool))

    random.shuffle(selected)
    return selected

direct_input_features = {
    "GPA": {"question": "What is your approximate GPA (0.0–4.0)?", "type": "number", "min": 0.0, "max": 4.0, "step": 0.1, "default": 3.0},
    "Years_of_Experience": {"question": "Years of professional experience:", "type": "number", "min": 0, "max": 50, "step": 1, "default": 0}
}

# ====================== STREAMLIT MAIN APP ======================
def main():
    apply_custom_css()
    st.title("🧭 Career Path Finder")
    st.markdown("Discover careers that match your unique strengths and preferences.")
    st.sidebar.title("About This Tool")
    st.sidebar.info("Answer a few questions to explore career paths that fit you best.")
    st.sidebar.write(f"🔍 Based on {len(data)} data samples")

    if 'user_responses' not in st.session_state:
        st.session_state.user_responses = {}
    if 'questions' not in st.session_state or st.session_state.get('reset_questions', False):
        st.session_state.questions = get_randomized_questions()
        st.session_state.reset_questions = False

    tab1, tab2 = st.tabs(["Take Assessment", "Career Insights"])

    with tab1:
        st.header("Career Compatibility Assessment")
        if st.session_state.user_responses:
            if st.button("🔄 Start New Assessment"):
                st.session_state.user_responses = {}
                st.session_state.reset_questions = True
                st.experimental_rerun()

        st.write("Answer these to receive career suggestions.")

        with st.expander("Your Background"):
            for feature, config in direct_input_features.items():
                st.session_state.user_responses[feature] = st.number_input(
                    config["question"],
                    min_value=config["min"],
                    max_value=config["max"],
                    value=config["default"],
                    step=config["step"],
                    key=f"num_{feature}"
                )

        st.subheader("Personality and Preferences")
        for i, q in enumerate(st.session_state.questions):
            selected_option = st.radio(
                q["question"],
                q["options"],
                index=0,
                key=f"q_{i}"
            )
            st.session_state.user_responses[q["feature"]] = selected_option

        if st.button("🎯 Get Career Matches"):
            input_df = pd.DataFrame([st.session_state.user_responses])
            encoded_input = input_df.copy()

            # Encoding user input like training data
            for col in encoded_input.columns:
                if encoded_input[col].dtype == 'object':
                    encoded_input[col] = LabelEncoder().fit(data[col].astype(str)).transform(encoded_input[col].astype(str))

            prediction = model.predict(encoded_input)[0]
            predicted_career = target_le.inverse_transform([prediction])[0]

            st.success(f"🔮 Top Career Match: **{predicted_career}**")

            similar = data[data['Predicted_Career_Field'] == prediction].sample(n=min(5, len(data)), random_state=42)
            st.subheader("📌 Sample Profiles in This Field")
            st.dataframe(similar.drop(columns=['Predicted_Career_Field']), use_container_width=True)

    with tab2:
        st.header("📊 Career Field Distribution")
        top_fields = data['Predicted_Career_Field'].value_counts().head(10)

        fig, ax = plt.subplots()
        top_fields.plot(kind='barh', ax=ax, color='skyblue')
        ax.set_xlabel("Count")
        ax.set_ylabel("Career")
        ax.set_title("Top Career Fields")
        st.pyplot(fig)

        st.subheader("📄 Sample Data")
        st.dataframe(data.head(20), use_container_width=True)

if __name__ == "__main__":
    main()
