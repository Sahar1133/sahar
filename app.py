import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import base64

# Set page config
st.set_page_config(
    page_title="AI Career Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Data loading and cleaning
@st.cache_data
def load_data():
    # Load the dataset
    data = pd.read_csv("new_updated_data.csv")
    
    # Data cleaning
    # Fix salary range formatting
    data['Salary_Expectation'] = data['Salary_Expectation'].str.replace('‰ŰŇ', '-')
    
    # Convert GPA to numeric (handle any non-numeric values)
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

# Preprocessing
def preprocess_data(data):
    # Encode categorical variables
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        if col != 'Predicted_Career_Field':
            data[col] = le.fit_transform(data[col])
    
    # Encode target variable
    data['Predicted_Career_Field'] = le.fit_transform(data['Predicted_Career_Field'])
    
    return data, le

processed_data, target_le = preprocess_data(data.copy())

# Train the model
def train_model(data):
    X = data.drop('Predicted_Career_Field', axis=1)
    y = data['Predicted_Career_Field']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

model, accuracy = train_model(processed_data)

# Feature importance
def get_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create a DataFrame for visualization
    feature_importance = pd.DataFrame({
        'Feature': [features[i] for i in indices],
        'Importance': importances[indices]
    })
    
    return feature_importance

features = processed_data.drop('Predicted_Career_Field', axis=1).columns
feature_importance = get_feature_importance(model, features)

# Question templates
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
        },
        {
            "question": "How do you handle team disagreements?",
            "options": [
                {"text": "Avoid getting involved", "value": "Low"},
                {"text": "Try to find middle ground", "value": "Medium"},
                {"text": "Facilitate constructive resolution", "value": "High"}
            ]
        }
    ],
    "Communication_Skills": [
        {
            "question": "How comfortable are you presenting ideas to a group?",
            "options": [
                {"text": "Very uncomfortable", "value": "Low"},
                {"text": "Somewhat comfortable", "value": "Medium"},
                {"text": "Very comfortable", "value": "High"}
            ]
        },
        {
            "question": "When explaining complex topics, you:",
            "options": [
                {"text": "Struggle to simplify", "value": "Low"},
                {"text": "Can explain with some effort", "value": "Medium"},
                {"text": "Can break down effectively", "value": "High"}
            ]
        }
    ],
    "Leadership_Skills": [
        {
            "question": "In leadership roles, you tend to:",
            "options": [
                {"text": "Avoid taking charge", "value": "Low"},
                {"text": "Lead when necessary", "value": "Medium"},
                {"text": "Naturally take initiative", "value": "High"}
            ]
        },
        {
            "question": "When delegating tasks, you:",
            "options": [
                {"text": "Prefer to do things yourself", "value": "Low"},
                {"text": "Delegate with some guidance", "value": "Medium"},
                {"text": "Effectively assign based on strengths", "value": "High"}
            ]
        }
    ],
    "Problem_Solving": [
        {
            "question": "When faced with a complex problem, you:",
            "options": [
                {"text": "Struggle to find solutions", "value": "Low"},
                {"text": "Can solve with time/help", "value": "Medium"},
                {"text": "Enjoy the challenge and find solutions", "value": "High"}
            ]
        }
    ],
    "Creativity_Score": [
        {
            "question": "How would you describe your creativity?",
            "options": [
                {"text": "Prefer structured approaches", "value": "Low"},
                {"text": "Somewhat creative", "value": "Medium"},
                {"text": "Highly innovative", "value": "High"}
            ]
        }
    ],
    "Stress_Tolerance": [
        {
            "question": "Under pressure, you typically:",
            "options": [
                {"text": "Become overwhelmed", "value": "Low"},
                {"text": "Manage with some difficulty", "value": "Medium"},
                {"text": "Remain calm and focused", "value": "High"}
            ]
        }
    ]
}

# Direct input features
direct_input_features = {
    "GPA": {
        "type": "number",
        "min": 0.0,
        "max": 4.0,
        "step": 0.01,
        "default": 3.0
    },
    "Years_of_Experience": {
        "type": "number",
        "min": 0,
        "max": 50,
        "step": 1,
        "default": 5
    },
    "Certifications_Count": {
        "type": "number",
        "min": 0,
        "max": 20,
        "step": 1,
        "default": 2
    }
}

# Streamlit app
def main():
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
    
    st.sidebar.title("Model Performance")
    st.sidebar.write(f"Accuracy: {accuracy:.2%}")
    
    with st.expander("Feature Importance"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), ax=ax)
        ax.set_title('Top 15 Important Features for Career Prediction')
        st.pyplot(fig)
    
    tab1, tab2 = st.tabs(["Career Prediction", "Dataset Overview"])
    
    with tab1:
        st.header("Career Prediction Questionnaire")
        st.markdown('<p class="big-font">Please answer the following questions to get your career prediction:</p>', unsafe_allow_html=True)
        
        user_responses = {}
        
        # Direct input features
        with st.expander("Academic & Professional Information"):
            for feature, config in direct_input_features.items():
                if config["type"] == "number":
                    user_responses[feature] = st.number_input(
                        f"Your {feature.replace('_', ' ')}:",
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["default"],
                        step=config["step"]
                    )
        
        # Questionnaire features
        for feature, question_list in questions.items():
            with st.expander(feature.replace("_", " ")):
                for i, question in enumerate(question_list):
                    options = [opt["text"] for opt in question["options"]]
                    values = [opt["value"] for opt in question["options"]]
                    
                    selected_option = st.radio(
                        question["question"],
                        options,
                        key=f"{feature}_{i}"
                    )
                    
                    # Store the corresponding value
                    selected_value = values[options.index(selected_option)]
                    if feature not in user_responses:
                        user_responses[feature] = []
                    user_responses[feature].append(selected_value)
        
        # Convert responses to model input format
        def prepare_input(responses):
            # Create a DataFrame with all features initialized to median/mean
            input_data = processed_data.drop('Predicted_Career_Field', axis=1).iloc[0:1].copy()
            
            # Reset all values to median (for numerical) or mode (for categorical)
            for col in input_data.columns:
                if col in direct_input_features:
                    input_data[col] = responses[col]
                elif col in questions:
                    # For question-based features, take the most frequent response level
                    levels = [level for level in responses[col] if level in ['Low', 'Medium', 'High']]
                    if levels:
                        # Convert to numerical (assuming same encoding as during training)
                        level_counts = {'Low': 0, 'Medium': 1, 'High': 2}
                        avg_level = np.mean([level_counts[level] for level in levels])
                        input_data[col] = avg_level
                else:
                    # For other features, use median (numerical) or mode (categorical)
                    if pd.api.types.is_numeric_dtype(input_data[col]):
                        input_data[col] = processed_data[col].median()
                    else:
                        input_data[col] = processed_data[col].mode()[0]
            
            return input_data
        
        if st.button("Predict My Career"):
            if len(user_responses) < 5:  # Basic validation
                st.warning("Please answer more questions for a better prediction.")
            else:
                with st.spinner("Analyzing your responses..."):
                    input_data = prepare_input(user_responses)
                    prediction = model.predict(input_data)
                    predicted_career = target_le.inverse_transform(prediction)[0]
                    
                    # Get explanation
                    tree_rules = export_text(model, feature_names=list(input_data.columns))
                    
                    st.success(f"Predicted Career: **{predicted_career}**")
                    
                    with st.expander("Why this prediction?"):
                        st.write("The prediction is based on the following key factors from your responses:")
                        
                        # Get top 3 influential features for this prediction
                        input_series = input_data.iloc[0]
                        feature_contributions = []
                        
                        for feature in feature_importance['Feature'].head(5):
                            if feature in input_series.index:
                                value = input_series[feature]
                                feature_contributions.append((
                                    feature.replace('_', ' '),
                                    value,
                                    feature_importance[feature_importance['Feature'] == feature]['Importance'].values[0]
                                ))
                        
                        # Sort by importance
                        feature_contributions.sort(key=lambda x: x[2], reverse=True)
                        
                        for feat, val, imp in feature_contributions[:3]:
                            st.write(f"- **{feat}**: Your input was {val:.2f} (relative importance: {imp:.2f})")
                        
                        st.write("\n**Decision Path:**")
                        st.text(tree_rules.split('\n')[0])  # Show first few rules
                    
                    # Show similar careers
                    similar_careers = data['Predicted_Career_Field'].value_counts().index[:3]
                    st.info(f"Other careers you might consider: {', '.join(similar_careers)}")
    
    with tab2:
        st.header("Dataset Overview")
        st.write("This is the data used to train the career prediction model.")
        
        if st.checkbox("Show raw data"):
            st.dataframe(data)
        
        st.subheader("Career Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        data['Predicted_Career_Field'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Distribution of Predicted Careers")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
        st.subheader("Key Statistics")
        st.write(data.describe())

if __name__ == "__main__":
    main()