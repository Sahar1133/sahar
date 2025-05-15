import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Set page config with custom styling
st.set_page_config(
    page_title="AI Career Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS directly
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #f5f5f5;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 5px;
        padding: 8px;
    }
    
    /* Radio button styling */
    .stRadio>div {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .stRadio>div>label {
        margin-bottom: 5px;
        padding: 8px;
        border-radius: 5px;
        background-color: #f0f0f0;
    }
    
    .stRadio>div>label:hover {
        background-color: #e0e0e0;
    }
    
    /* Header styling */
    h1 {
        color: #2c3e50;
    }
    
    h2 {
        color: #3498db;
    }
    
    /* Success message */
    .stAlert {
        border-radius: 5px;
    }
    
    /* Expander styling */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    
    .stExpander>div>div>div>div {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Data loading and cleaning
@st.cache_data
def load_data():
    # Create sample data if file not found
    try:
        data = pd.read_csv("new_updated_data.xlsx")
    except FileNotFoundError:
        st.warning("Sample data not found. Using demo data.")
        data = pd.DataFrame({
            'Interest': ['Technology', 'Business', 'Arts'],
            'Work_Style': ['Independent', 'Collaborative', 'Flexible'],
            'Strengths': ['Analytical', 'Creative', 'Strategic'],
            'Communication_Skills': ['Medium', 'High', 'Low'],
            'Leadership_Skills': ['Medium', 'High', 'Low'],
            'Teamwork_Skills': ['High', 'Medium', 'Low'],
            'GPA': [3.5, 3.8, 3.2],
            'Years_of_Experience': [5, 10, 2],
            'Predicted_Career_Field': ['Software Developer', 'Marketing Manager', 'Graphic Designer']
        })
    
    # Data cleaning
    if 'Salary_Expectation' in data.columns:
        data['Salary_Expectation'] = data['Salary_Expectation'].str.replace('‰ŰŇ', '-')
    
    if 'GPA' in data.columns:
        data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
    
    # Fill missing values
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        data[col].fillna(data[col].median(), inplace=True)
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    return data

data = load_data()

# Rest of your code remains the same from the previous version...
# [Include all the remaining functions and code from the previous version]

def main():
    apply_custom_css()  # Apply CSS styling
    
    st.title("🧠 AI Career Prediction System")
    st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("About")
    st.sidebar.info("""
    This AI-powered career prediction system analyzes your skills, preferences, 
    and personality traits to suggest the most suitable career paths for you.
    """)
    
    # Rest of your main function remains the same...
    # [Include all the remaining code from the previous version's main() function]



def main():
    apply_custom_css()
    
    st.title("🧠 AI Career Prediction System")
    
    st.sidebar.title("About")
    st.sidebar.info("This AI-powered career prediction system analyzes your skills and preferences.")
    
    # Initialize session state for user responses if not already present
    if 'user_responses' not in st.session_state:
        st.session_state.user_responses = {}
    
    # Show questionnaire regardless of data loading
    st.header("Career Prediction Questionnaire")
    
    user_responses = {}
    
    # Direct input features
    with st.expander("Academic & Professional Information"):
        user_responses['GPA'] = st.number_input(
            "Your GPA:",
            min_value=0.0,
            max_value=4.0,
            value=3.0,
            step=0.1
        )
        user_responses['Years_of_Experience'] = st.number_input(
            "Your Years of Experience:",
            min_value=0,
            max_value=50,
            value=5,
            step=1
        )
    
    # Questionnaire sections
    with st.expander("Teamwork Skills"):
        teamwork = st.radio(
            "In group projects, you typically:",
            ["Work separately", "Coordinate some", "Actively collaborate"],
            key="teamwork1"
        )
        user_responses['Teamwork_Skills'] = teamwork
    
    with st.expander("Communication Skills"):
        communication = st.radio(
            "How comfortable are you presenting ideas to a group?",
            ["Very uncomfortable", "Somewhat comfortable", "Very comfortable"],
            key="communication1"
        )
        user_responses['Communication_Skills'] = communication
    
    if st.button("Get Career Prediction"):
        if len(user_responses) < 2:
            st.warning("Please answer at least the required questions")
        else:
            # Store responses
            st.session_state.user_responses = user_responses
            
            # Show a sample prediction (since we don't have real data)
            sample_careers = ["Software Developer", "Data Analyst", "Project Manager", 
                             "Marketing Specialist", "Graphic Designer"]
            predicted_career = np.random.choice(sample_careers)
            
            st.success(f"Based on your responses, a good career match might be: {predicted_career}")
            
            # Show explanation
            with st.expander("Why this prediction?"):
                st.write("Key factors in your prediction:")
                st.write(f"- Teamwork: {user_responses.get('Teamwork_Skills', 'Not specified')}")
                st.write(f"- Communication: {user_responses.get('Communication_Skills', 'Not specified')}")
                st.write(f"- GPA: {user_responses.get('GPA', 'Not specified')}")
                st.write(f"- Experience: {user_responses.get('Years_of_Experience', 'Not specified')} years")
                
                st.write("\nThis is a simplified demo. With the full dataset, the prediction would be more accurate.")
