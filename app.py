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
        data = pd.read_csv("new_updated_data.csv")
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

if __name__ == "__main__":
    main()
