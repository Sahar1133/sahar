import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import random

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
        data = pd.read_excel("new_updated_data.xlsx")
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
    # Only encode columns that exist in the dataframe and are object type
    object_cols = [col for col in data.select_dtypes(include=['object']).columns 
                  if col in data.columns]
    
    for col in object_cols:
        if col != 'Predicted_Career_Field':
            data[col] = le.fit_transform(data[col].astype(str))
    
    if 'Predicted_Career_Field' in data.columns:
        data['Predicted_Career_Field'] = le.fit_transform(data['Predicted_Career_Field'])
    return data, le

processed_data, target_le = preprocess_data(data.copy())

def train_model(data):
    if 'Predicted_Career_Field' not in data.columns:
        st.error("Target column 'Predicted_Career_Field' not found in data")
        return None, 0
    
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
all_questions = {
    "Interest": [
        {
            "question": "Which of these subjects interests you most?",
            "options": [
                {"text": "Technology and Computers", "value": "Technology"},
                {"text": "Business and Finance", "value": "Business"},
                {"text": "Arts and Design", "value": "Arts"},
                {"text": "Engineering and Construction", "value": "Engineering"},
                {"text": "Healthcare and Medicine", "value": "Medical"}
            ]
        },
        {
            "question": "In your free time, you prefer to:",
            "options": [
                {"text": "Learn new tech skills", "value": "Technology"},
                {"text": "Read business news", "value": "Business"},
                {"text": "Create art or design", "value": "Arts"},
                {"text": "Build or fix things", "value": "Engineering"},
                {"text": "Help others with health advice", "value": "Medical"}
            ]
        },
        {
            "question": "Which type of books/magazines do you prefer?",
            "options": [
                {"text": "Tech journals and programming books", "value": "Technology"},
                {"text": "Business and finance publications", "value": "Business"},
                {"text": "Art and design magazines", "value": "Arts"},
                {"text": "Engineering manuals", "value": "Engineering"},
                {"text": "Medical journals", "value": "Medical"}
            ]
        }
    ],
    "Work_Style": [
        {
            "question": "Your ideal work environment is:",
            "options": [
                {"text": "Working alone with clear tasks", "value": "Independent"},
                {"text": "Working closely with a team", "value": "Collaborative"},
                {"text": "A mix of both with flexibility", "value": "Flexible"}
            ]
        },
        {
            "question": "When facing a complex problem, you:",
            "options": [
                {"text": "Prefer to solve it yourself", "value": "Independent"},
                {"text": "Brainstorm with colleagues", "value": "Collaborative"},
                {"text": "Depends on the situation", "value": "Flexible"}
            ]
        },
        {
            "question": "Your preferred work schedule is:",
            "options": [
                {"text": "Strict 9-5 with clear boundaries", "value": "Independent"},
                {"text": "Flexible hours with team coordination", "value": "Collaborative"},
                {"text": "Mix of structured and flexible time", "value": "Flexible"}
            ]
        }
    ],
    "Strengths": [
        {
            "question": "Your strongest skill is:",
            "options": [
                {"text": "Analyzing data and patterns", "value": "Analytical"},
                {"text": "Coming up with new ideas", "value": "Creative"},
                {"text": "Planning long-term strategies", "value": "Strategic"},
                {"text": "Hands-on problem solving", "value": "Practical"}
            ]
        },
        {
            "question": "You're particularly good at:",
            "options": [
                {"text": "Math and logical reasoning", "value": "Analytical"},
                {"text": "Artistic expression", "value": "Creative"},
                {"text": "Seeing the big picture", "value": "Strategic"},
                {"text": "Building physical solutions", "value": "Practical"}
            ]
        },
        {
            "question": "In school, you excelled at:",
            "options": [
                {"text": "Math and science subjects", "value": "Analytical"},
                {"text": "Creative writing and arts", "value": "Creative"},
                {"text": "History and social studies", "value": "Strategic"},
                {"text": "Shop class and hands-on projects", "value": "Practical"}
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
        },
        {
            "question": "In meetings, you typically:",
            "options": [
                {"text": "Rarely speak up", "value": "Low"},
                {"text": "Contribute when asked", "value": "Medium"},
                {"text": "Frequently share ideas", "value": "High"}
            ]
        },
        {
            "question": "When explaining complex topics, you:",
            "options": [
                {"text": "Struggle to explain clearly", "value": "Low"},
                {"text": "Can explain with some effort", "value": "Medium"},
                {"text": "Explain clearly and effectively", "value": "High"}
            ]
        }
    ],
    "Leadership_Skills": [
        {
            "question": "When assigned to lead a project, you:",
            "options": [
                {"text": "Feel anxious about the responsibility", "value": "Low"},
                {"text": "Manage but prefer not to lead", "value": "Medium"},
                {"text": "Feel confident in your ability", "value": "High"}
            ]
        },
        {
            "question": "Your leadership style is:",
            "options": [
                {"text": "Avoid leadership roles", "value": "Low"},
                {"text": "Lead when necessary", "value": "Medium"},
                {"text": "Naturally take charge", "value": "High"}
            ]
        },
        {
            "question": "When conflicts arise in a team, you:",
            "options": [
                {"text": "Avoid getting involved", "value": "Low"},
                {"text": "Help mediate if asked", "value": "Medium"},
                {"text": "Proactively resolve conflicts", "value": "High"}
            ]
        }
    ],
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
        },
        {
            "question": "Your approach to team decisions is:",
            "options": [
                {"text": "Prefer others to decide", "value": "Low"},
                {"text": "Contribute but go with majority", "value": "Medium"},
                {"text": "Actively shape team decisions", "value": "High"}
            ]
        }
    ],
    "Problem_Solving": [
        {
            "question": "When facing a new problem, you:",
            "options": [
                {"text": "Follow established procedures", "value": "Low"},
                {"text": "Combine known methods", "value": "Medium"},
                {"text": "Invent new approaches", "value": "High"}
            ]
        },
        {
            "question": "Your problem-solving style is:",
            "options": [
                {"text": "Methodical and step-by-step", "value": "Low"},
                {"text": "Balanced between creative and logical", "value": "Medium"},
                {"text": "Highly innovative and unconventional", "value": "High"}
            ]
        }
    ]
}

direct_input_features = {
    "GPA": {"type": "number", "min": 0.0, "max": 4.0, "step": 0.1, "default": 3.0},
    "Years_of_Experience": {"type": "number", "min": 0, "max": 50, "step": 1, "default": 5}
}

def select_random_questions(all_questions, questions_per_category=2):
    """Select random questions from each category"""
    selected_questions = {}
    for category, question_list in all_questions.items():
        selected_questions[category] = random.sample(question_list, min(questions_per_category, len(question_list)))
    return selected_questions

# ====================== STREAMLIT APP ======================
def main():
    apply_custom_css()
    
    st.title("🧠 AI Career Prediction System")
    st.markdown("Discover your ideal career path based on your skills and preferences.")
    
    # Initialize session state for questions if not already present
    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = select_random_questions(all_questions)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("This tool uses machine learning to match your profile with suitable careers.")
    st.sidebar.write(f"Model Accuracy: **{accuracy:.1%}**")
    
    # Add button to reshuffle questions
    if st.sidebar.button("🔄 Get New Questions"):
        st.session_state.selected_questions = select_random_questions(all_questions)
        st.rerun()
    
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
        for feature, question_list in st.session_state.selected_questions.items():
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
                    
                    # Create label encoders for categorical features
                    le_dict = {}
                    for col in data.select_dtypes(include=['object']).columns:
                        if col in data.columns and col != 'Predicted_Career_Field':
                            le = LabelEncoder()
                            le.fit(data[col].astype(str))
                            le_dict[col] = le
                    
                    for col in input_data.columns:
                        if col in user_responses:
                            if isinstance(user_responses[col], list):  # For question-based features
                                if col in ['Communication_Skills', 'Leadership_Skills', 'Teamwork_Skills', 'Problem_Solving']:
                                    # Handle Low/Medium/High scale
                                    level_map = {"Low": 0, "Medium": 1, "High": 2}
                                    avg_level = np.mean([level_map[val] for val in user_responses[col]])
                                    input_data[col] = avg_level
                                else:
                                    # For other categorical features, take the first response and encode it
                                    if col in le_dict:
                                        input_data[col] = le_dict[col].transform([user_responses[col][0]])[0]
                            else:  # For direct inputs (numerical)
                                input_data[col] = user_responses[col]
                        else:
                            input_data[col] = processed_data[col].median()
                    
                    # Make prediction
                    try:
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
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")
    
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
