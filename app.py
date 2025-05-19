# ====================== IMPORTS ======================
import matplotlib.pyplot as plt # For data visualization
import pandas as pd # For data manipulation and analysis
import numpy as np # For numerical operations
import streamlit as st # For building the web app interface
from sklearn.tree import DecisionTreeClassifier, export_text # Machine learning model
from sklearn.model_selection import train_test_split # For splitting data into train/test sets
from sklearn.preprocessing import LabelEncoder # For encoding categorical variables
from sklearn.metrics import accuracy_score # For evaluating model performance
import random # For randomizing questions

# ====================== STYLING & SETUP ======================
# Configure the Streamlit page settings
st.set_page_config(
    page_title="AI Powered Career Prediction Based on Personality Traits", # Browser tab title
    page_icon="🧭", # Browser tab icon
    layout="wide", # Use wider page layout
    initial_sidebar_state="expanded" # Start with sidebar expanded
)

def apply_custom_css():
    """Applies custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
    /* Main background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    /* Card styling */
    .stCard {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e1e4e8;
    }
    
    /* Button styling with animation */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Input fields with modern look */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
        background: #f8fafc;
        transition: all 0.3s;
    }
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Radio buttons with card-like appearance */
    .stRadio > div {
        flex-direction: column;
        gap: 12px;
    }
    .stRadio > div > label {
        background: white;
        padding: 16px;
        border-radius: 10px;
        transition: all 0.2s;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .stRadio > div > label:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    }
    .stRadio > div > label[data-baseweb="radio"]:first-child {
        margin-top: 0;
    }
    
    /* Headers with modern typography */
    h1 {
        color: #2d3748;
        font-weight: 700;
        margin-bottom: 1rem;
        position: relative;
    }
    h1:after {
        content: "";
        position: absolute;
        bottom: -8px;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    h2 {
        color: #4a5568;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    h3 {
        color: #4a5568;
        font-weight: 500;
    }
    
    /* Expanders with card styling */
    .stExpander {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    .stExpander > summary {
        font-weight: 600;
        padding: 1rem 1.5rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #667eea;
        font-weight: 600;
    }
    .stTabs [aria-selected="false"] {
        background-color: #f8fafc;
        color: #4a5568;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: white;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255,255,255,0.05);
        color: white;
        border-color: rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #a0aec0;
    }
    </style>
    """, unsafe_allow_html=True)

# ====================== DATA LOADING & PREPROCESSING ======================
@st.cache_data # Cache the data to avoid reloading on every interaction
def load_data():
    career_options = [
        # List of potential career options
        'Software Developer', 'Data Scientist', 'AI Engineer', 
        'Cybersecurity Specialist', 'Cloud Architect',
        'Marketing Manager', 'Financial Analyst', 'HR Manager',
        'Entrepreneur', 'Investment Banker',
        'Graphic Designer', 'Video Editor', 'Music Producer',
        'Creative Writer', 'Art Director',
        'Mechanical Engineer', 'Electrical Engineer', 
        'Civil Engineer', 'Robotics Engineer',
        'Doctor', 'Nurse', 'Psychologist', 
        'Physical Therapist', 'Medical Researcher',
        'Biotechnologist', 'Research Scientist', 
        'Environmental Scientist', 'Physicist',
        'Teacher', 'Professor', 'Educational Consultant',
        'Curriculum Developer',
        'Lawyer', 'Judge', 'Legal Consultant',
        'UX Designer', 'Product Manager',
        'Journalist', 'Public Relations Specialist',
        'Architect', 'Urban Planner',
        'Chef', 'Event Planner', 'Fashion Designer'
    ]
    
    try:
        # Try to load real dataset
        data = pd.read_excel("new_updated_data.xlsx")
        # If career field data is sparse, generate random career assignments
        if len(data['Predicted_Career_Field'].unique()) < 20:
            data['Predicted_Career_Field'] = np.random.choice(career_options, size=len(data))
    except FileNotFoundError:
        # Fallback to demo data if real dataset not found
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
    # Clean GPA data if it exists
    if 'GPA' in data.columns:
        data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
        data['GPA'].fillna(data['GPA'].median(), inplace=True)
    
    return data

data = load_data()

# ====================== MODEL TRAINING ======================
def preprocess_data(data):
    le = LabelEncoder()
    # Identify categorical columns (excluding the target)
    object_cols = [col for col in data.select_dtypes(include=['object']).columns 
                  if col in data.columns]
    # Encode each categorical column
    for col in object_cols:
        if col != 'Predicted_Career_Field':
            data[col] = le.fit_transform(data[col].astype(str))
    # Encode the target variable (career field)
    if 'Predicted_Career_Field' in data.columns:
        data['Predicted_Career_Field'] = le.fit_transform(data['Predicted_Career_Field'])
    return data, le # Return processed data and the target encoder
# Process the data
processed_data, target_le = preprocess_data(data.copy())

# ====================== MODEL TRAINING ======================
def train_model(data):
    if 'Predicted_Career_Field' not in data.columns:
        st.error("Target column not found in data")
        return None, 0
    # Prepare features (X) and target (y)
    X = data.drop('Predicted_Career_Field', axis=1)
    y = data['Predicted_Career_Field']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the decision tree model
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy
    
# Train the model
model, accuracy = train_model(processed_data)

# ====================== QUESTIONNAIRE ======================
def get_all_questions():
    """Returns a pool of 30 questions"""
    return [
        # Interest Questions (1-8)
        {
            "question": "1. Which of these activities excites you most?",
            "options": [
                {"text": "Coding or working with technology", "value": "Technology"},
                {"text": "Analyzing market trends", "value": "Business"},
                {"text": "Creating art or designs", "value": "Arts"},
                {"text": "Building or fixing mechanical things", "value": "Engineering"},
                {"text": "Helping people with health issues", "value": "Medical"},
                {"text": "Conducting experiments", "value": "Science"},
                {"text": "Teaching others", "value": "Education"},
                {"text": "Debating or solving legal problems", "value": "Law"}
            ],
            "feature": "Interest"
        },
        {
            "question": "2. What type of books/movies do you enjoy most?",
            "options": [
                {"text": "Sci-fi and technology", "value": "Technology"},
                {"text": "Business success stories", "value": "Business"},
                {"text": "Creative arts and design", "value": "Arts"},
                {"text": "How things work", "value": "Engineering"},
                {"text": "Medical dramas", "value": "Medical"},
                {"text": "Scientific discoveries", "value": "Science"},
                {"text": "Educational content", "value": "Education"},
                {"text": "Courtroom dramas", "value": "Law"}
            ],
            "feature": "Interest"
        },
        {
            "question": "3. Which subject did you enjoy most in school?",
            "options": [
                {"text": "Computer Science", "value": "Technology"},
                {"text": "Economics/Business", "value": "Business"},
                {"text": "Art/Music", "value": "Arts"},
                {"text": "Physics/Engineering", "value": "Engineering"},
                {"text": "Biology/Medicine", "value": "Medical"},
                {"text": "Chemistry/Physics", "value": "Science"},
                {"text": "Education/Psychology", "value": "Education"},
                {"text": "Government/Law", "value": "Law"}
            ],
            "feature": "Interest"
        },
        {
            "question": "4. What type of projects do you enjoy most?",
            "options": [
                {"text": "Developing software/apps", "value": "Technology"},
                {"text": "Creating business plans", "value": "Business"},
                {"text": "Designing visuals/artwork", "value": "Arts"},
                {"text": "Building physical prototypes", "value": "Engineering"},
                {"text": "Helping people directly", "value": "Medical"},
                {"text": "Research experiments", "value": "Science"},
                {"text": "Teaching/mentoring", "value": "Education"},
                {"text": "Analyzing legal cases", "value": "Law"}
            ],
            "feature": "Interest"
        },

        # Work Style Questions (5-8)
        {
            "question": "5. How do you prefer to work?",
            "options": [
                {"text": "Alone with clear tasks", "value": "Independent"},
                {"text": "In a team environment", "value": "Collaborative"},
                {"text": "A flexible mix of both", "value": "Flexible"}
            ],
            "feature": "Work_Style"
        },
        {
            "question": "6. Your ideal project would involve:",
            "options": [
                {"text": "Working independently on your part", "value": "Independent"},
                {"text": "Constant collaboration with others", "value": "Collaborative"},
                {"text": "Some teamwork with independent phases", "value": "Flexible"}
            ],
            "feature": "Work_Style"
        },
        {
            "question": "7. When facing a difficult problem, you:",
            "options": [
                {"text": "Prefer to solve it yourself", "value": "Independent"},
                {"text": "Ask colleagues for input", "value": "Collaborative"},
                {"text": "Try yourself first, then ask for help", "value": "Flexible"}
            ],
            "feature": "Work_Style"
        },
        {
            "question": "8. Your ideal work schedule would be:",
            "options": [
                {"text": "Strict 9-5 with clear boundaries", "value": "Independent"},
                {"text": "Flexible hours with team coordination", "value": "Collaborative"},
                {"text": "Mix of structured and flexible time", "value": "Flexible"}
            ],
            "feature": "Work_Style"
        },

        # Strengths Questions (9-12)
        {
            "question": "9. What comes most naturally to you?",
            "options": [
                {"text": "Solving complex problems", "value": "Analytical"},
                {"text": "Coming up with creative ideas", "value": "Creative"},
                {"text": "Planning long-term strategies", "value": "Strategic"},
                {"text": "Building practical solutions", "value": "Practical"}
            ],
            "feature": "Strengths"
        },
        {
            "question": "10. Others would describe you as:",
            "options": [
                {"text": "Logical and detail-oriented", "value": "Analytical"},
                {"text": "Imaginative and original", "value": "Creative"},
                {"text": "Visionary and forward-thinking", "value": "Strategic"},
                {"text": "Hands-on and resourceful", "value": "Practical"}
            ],
            "feature": "Strengths"
        },
        {
            "question": "11. Your strongest skill is:",
            "options": [
                {"text": "Data analysis", "value": "Analytical"},
                {"text": "Creative thinking", "value": "Creative"},
                {"text": "Long-term planning", "value": "Strategic"},
                {"text": "Practical implementation", "value": "Practical"}
            ],
            "feature": "Strengths"
        },
        {
            "question": "12. In school projects, you typically:",
            "options": [
                {"text": "Handled the data/analysis parts", "value": "Analytical"},
                {"text": "Came up with creative concepts", "value": "Creative"},
                {"text": "Organized the overall strategy", "value": "Strategic"},
                {"text": "Built the practical deliverables", "value": "Practical"}
            ],
            "feature": "Strengths"
        },

        # Communication Skills (13-16)
        {
            "question": "13. In social situations, you:",
            "options": [
                {"text": "Prefer listening to speaking", "value": "Low"},
                {"text": "Speak when you have something to say", "value": "Medium"},
                {"text": "Easily engage in conversations", "value": "High"}
            ],
            "feature": "Communication_Skills"
        },
        {
            "question": "14. When explaining something complex, you:",
            "options": [
                {"text": "Struggle to put it in simple terms", "value": "Low"},
                {"text": "Can explain if you prepare", "value": "Medium"},
                {"text": "Naturally simplify complex ideas", "value": "High"}
            ],
            "feature": "Communication_Skills"
        },
        {
            "question": "15. In group presentations, you typically:",
            "options": [
                {"text": "Handle the background research", "value": "Low"},
                {"text": "Present your specific part", "value": "Medium"},
                {"text": "Take the lead in presenting", "value": "High"}
            ],
            "feature": "Communication_Skills"
        },
        {
            "question": "16. When networking professionally, you:",
            "options": [
                {"text": "Find it challenging", "value": "Low"},
                {"text": "Can do it when necessary", "value": "Medium"},
                {"text": "Enjoy meeting new people", "value": "High"}
            ],
            "feature": "Communication_Skills"
        },

        # Leadership Skills (17-20)
        {
            "question": "17. When a group needs direction, you:",
            "options": [
                {"text": "Wait for someone else to step up", "value": "Low"},
                {"text": "Help if no one else does", "value": "Medium"},
                {"text": "Naturally take the lead", "value": "High"}
            ],
            "feature": "Leadership_Skills"
        },
        {
            "question": "18. Your approach to responsibility is:",
            "options": [
                {"text": "Avoid taking charge", "value": "Low"},
                {"text": "Take charge when needed", "value": "Medium"},
                {"text": "Seek leadership roles", "value": "High"}
            ],
            "feature": "Leadership_Skills"
        },
        {
            "question": "19. In team projects, you usually:",
            "options": [
                {"text": "Follow others' lead", "value": "Low"},
                {"text": "Share leadership duties", "value": "Medium"},
                {"text": "Organize the team's work", "value": "High"}
            ],
            "feature": "Leadership_Skills"
        },
        {
            "question": "20. When making group decisions, you:",
            "options": [
                {"text": "Go with the majority", "value": "Low"},
                {"text": "Voice your opinion when asked", "value": "Medium"},
                {"text": "Facilitate the decision-making", "value": "High"}
            ],
            "feature": "Leadership_Skills"
        },

        # Teamwork Skills (21-24)
        {
            "question": "21. In group settings, you usually:",
            "options": [
                {"text": "Focus on your individual tasks", "value": "Low"},
                {"text": "Coordinate when necessary", "value": "Medium"},
                {"text": "Actively collaborate with others", "value": "High"}
            ],
            "feature": "Teamwork_Skills"
        },
        {
            "question": "22. When a teammate needs help, you:",
            "options": [
                {"text": "Let them figure it out", "value": "Low"},
                {"text": "Help if they ask", "value": "Medium"},
                {"text": "Proactively offer assistance", "value": "High"}
            ],
            "feature": "Teamwork_Skills"
        },
        {
            "question": "23. Your view on teamwork is:",
            "options": [
                {"text": "Prefer working alone", "value": "Low"},
                {"text": "Teamwork has its benefits", "value": "Medium"},
                {"text": "Believe in the power of collaboration", "value": "High"}
            ],
            "feature": "Teamwork_Skills"
        },
        {
            "question": "24. In conflict situations, you:",
            "options": [
                {"text": "Avoid getting involved", "value": "Low"},
                {"text": "Help mediate if needed", "value": "Medium"},
                {"text": "Actively work to resolve conflicts", "value": "High"}
            ],
            "feature": "Teamwork_Skills"
        },

        # Additional Career-Relevant Questions (25-30)
        {
            "question": "25. How do you handle deadlines?",
            "options": [
                {"text": "I often procrastinate", "value": "Low"},
                {"text": "I meet them with some effort", "value": "Medium"},
                {"text": "I consistently meet them early", "value": "High"}
            ],
            "feature": "Time_Management"
        },
        {
            "question": "26. When learning something new, you prefer:",
            "options": [
                {"text": "Hands-on practice", "value": "Practical"},
                {"text": "Theoretical understanding", "value": "Theoretical"},
                {"text": "Visual demonstrations", "value": "Visual"},
                {"text": "Group discussions", "value": "Social"}
            ],
            "feature": "Learning_Style"
        },
        {
            "question": "27. Your ideal work environment is:",
            "options": [
                {"text": "Structured and predictable", "value": "Structured"},
                {"text": "Dynamic and changing", "value": "Dynamic"},
                {"text": "Creative and open", "value": "Creative"},
                {"text": "Fast-paced and challenging", "value": "Challenging"}
            ],
            "feature": "Work_Environment"
        },
        {
            "question": "28. When faced with a problem, you:",
            "options": [
                {"text": "Follow established procedures", "value": "Procedural"},
                {"text": "Brainstorm creative solutions", "value": "Creative"},
                {"text": "Analyze data thoroughly", "value": "Analytical"},
                {"text": "Ask others for advice", "value": "Collaborative"}
            ],
            "feature": "Problem_Solving"
        },
        {
            "question": "29. When making decisions, you rely mostly on:",
            "options": [
                {"text": "Logic and facts", "value": "Logical"},
                {"text": "Gut feelings", "value": "Intuitive"},
                {"text": "Others' opinions", "value": "Social"},
                {"text": "Past experiences", "value": "Experiential"}
            ],
            "feature": "Decision_Making"
        },
        {
            "question": "30. You consider yourself more:",
            "options": [
                {"text": "Realistic and practical", "value": "Practical"},
                {"text": "Imaginative and innovative", "value": "Innovative"},
                {"text": "People-oriented", "value": "Social"},
                {"text": "Detail-oriented", "value": "Detail"}
            ],
            "feature": "Self_Perception"
        }
    ]

def get_randomized_questions():
    """Selects 10 random questions from the pool of 30."""
    all_questions = get_all_questions()
    features = list(set(q['feature'] for q in all_questions))
    selected = []

    # First pick one from each feature category
    for feature in features:
        feature_questions = [q for q in all_questions if q['feature'] == feature]
        if feature_questions:
            selected.append(random.choice(feature_questions))

    # Remove selected questions from the pool
    remaining = [q for q in all_questions if q not in selected]

    # Calculate how many more we need to reach 10
    needed = 10 - len(selected)

    # Only sample if we have remaining questions and need more
    if needed > 0 and remaining:
        selected.extend(random.sample(remaining, min(needed, len(remaining))))

    random.shuffle(selected)
    return selected

direct_input_features = {
    "GPA": {
        "question": "What is your approximate GPA (0.0-4.0)?",
        "type": "number", 
        "min": 0.0, 
        "max": 4.0, 
        "step": 0.1, 
        "default": 3.0
    },
    "Years_of_Experience": {
        "question": "Years of professional experience (if any):",
        "type": "number", 
        "min": 0, 
        "max": 50, 
        "step": 1, 
        "default": 0
    }
}

# ====================== STREAMLIT APP ======================
def main():
    apply_custom_css()
    
    # Initialize session state
    if 'user_responses' not in st.session_state:
        st.session_state.user_responses = {}
    if 'questions' not in st.session_state:
        st.session_state.questions = get_randomized_questions()

    # Set up page title and description
    st.title("🧭 AI Powered Career Prediction Based on Personality Traits")
    st.markdown("Discover careers that match your unique strengths and preferences.")

    # Sidebar information
    st.sidebar.title("About This Tool")
    st.sidebar.info("This assessment helps match your profile with suitable career options.")
    st.sidebar.write(f"*Based on analysis of {len(data)} career paths*")

    # Create two tabs for different functionalities
    tab1, tab2 = st.tabs(["Take Assessment", "Career Insights"])

    # Assessment Tab
    with tab1:
        st.header("Career Compatibility Assessment")
        st.write("Answer these questions to discover careers that fit your profile.")

        # Background information section
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

        # Display randomized questions
        st.subheader("Personality and Preferences")
        for i, q in enumerate(st.session_state.questions):
            selected_option = st.radio(
                q["question"],
                [opt["text"] for opt in q["options"]],
                key=f"q_{i}"
            )
            selected_value = q["options"][[opt["text"] for opt in q["options"]].index(selected_option)]["value"]
            st.session_state.user_responses[q["feature"]] = selected_value

        # Prediction button
        if st.button("🔮 Find My Career Match"):
            required_fields = list(direct_input_features.keys()) + ['Interest', 'Work_Style', 'Strengths']
            filled_fields = [field for field in required_fields if field in st.session_state.user_responses]
            
            if len(filled_fields) < 3:
                st.warning("Please answer at least 3 questions (including GPA and Experience) for better results.")
            else:
                with st.spinner("Analyzing your unique profile..."):
                    input_data = processed_data.drop('Predicted_Career_Field', axis=1).iloc[0:1].copy()
                    
                    le_dict = {}
                    for col in data.select_dtypes(include=['object']).columns:
                        if col in data.columns and col != 'Predicted_Career_Field':
                            le = LabelEncoder()
                            le.fit(data[col].astype(str))
                            le_dict[col] = le

                    # Map user responses to model input format
                    for col in input_data.columns:
                        if col in st.session_state.user_responses:
                            if col in ['Communication_Skills', 'Leadership_Skills', 'Teamwork_Skills']:
                                level_map = {"Low": 0, "Medium": 1, "High": 2}
                                input_data[col] = level_map.get(st.session_state.user_responses[col], 1)
                                # Encode categorical features
                            elif col in le_dict:
                                try:
                                    input_data[col] = le_dict[col].transform([st.session_state.user_responses[col]])[0]
                                except ValueError:
                                    input_data[col] = processed_data[col].mode()[0]
                            else:
                                # Direct numerical values
                                input_data[col] = st.session_state.user_responses[col]
                        else:
                            input_data[col] = processed_data[col].median()
                    
                    try:
                        # Make prediction
                        prediction = model.predict(input_data)
                        predicted_career = target_le.inverse_transform(prediction)[0]

    

                        # Display results
                        st.success(f"### Your Best Career Match: **{predicted_career}**")
                        
                        with st.expander("💡 Why this career matches you"):
                            st.write("This career aligns well with your profile because of:")
                            
                            feat_importances = pd.Series(model.feature_importances_, index=input_data.columns)
                            top_features = feat_importances.sort_values(ascending=False).head(3)
                            
                            for feat in top_features.index:
                                importance_desc = ""
                                if feat == "Interest":
                                    interest_val = st.session_state.user_responses.get("Interest", "Various")
                                    importance_desc = f"Your strong interest in {interest_val} fields"
                                elif feat == "Work_Style":
                                    style_val = st.session_state.user_responses.get("Work_Style", "Various")
                                    importance_desc = f"Your preference for {style_val.lower()} work environments"
                                elif feat == "Strengths":
                                    strength_val = st.session_state.user_responses.get("Strengths", "Various")
                                    importance_desc = f"Your {strength_val.lower()} strengths"
                                elif feat == "GPA":
                                    gpa_val = st.session_state.user_responses.get("GPA", 3.0)
                                    importance_desc = f"Your academic performance (GPA: {gpa_val})"
                                elif feat == "Years_of_Experience":
                                    exp_val = st.session_state.user_responses.get("Years_of_Experience", 0)
                                    importance_desc = f"Your professional experience ({exp_val} years)"
                                else:
                                    importance_desc = f"Your responses about {feat.replace('_', ' ').lower()}"
                                
                                st.write(f"- **{importance_desc}** (weight: {top_features[feat]:.2f})")
                            
                            st.write("\nThis career path typically requires these characteristics, which match well with your profile.")
                        
                        with st.expander("📚 Learn more about this career"):
                            career_data = data[data['Predicted_Career_Field'] == predicted_career]
                            if not career_data.empty:
                                st.write(f"**Typical profile for {predicted_career}:**")
                                cols = st.columns(3)
                                with cols[0]:
                                    if 'GPA' in career_data.columns and not career_data['GPA'].isnull().all():
                                        st.metric("Average GPA", f"{career_data['GPA'].mean():.1f}")
                                    else:
                                        st.metric("Average GPA", "N/A")
                                with cols[1]:
                                    if 'Years_of_Experience' in career_data.columns:
                                        st.metric("Avg. Experience", f"{career_data['Years_of_Experience'].mean():.1f} years")
                                    else:
                                        st.metric("Avg. Experience", "N/A")
                                with cols[2]:
                                    if 'Interest' in career_data.columns:
                                        st.metric("Common Interest", career_data['Interest'].mode()[0])
                                    else:
                                        st.metric("Common Interest", "N/A")
                                
                                st.write("\n**Common characteristics:**")
                                if 'Work_Style' in career_data.columns:
                                    st.write(f"- Work Style: {career_data['Work_Style'].mode()[0]}")
                                if 'Strengths' in career_data.columns:
                                    st.write(f"- Strengths: {career_data['Strengths'].mode()[0]}")
                                if 'Communication_Skills' in career_data.columns:
                                    st.write(f"- Communication: {career_data['Communication_Skills'].mode()[0]}")
                            else:
                                st.write("No additional information available for this career in our dataset.")
                        
                    except Exception as e:
                        st.error(f"We encountered an issue analyzing your profile. Please try again.")
    
    with tab2:
        st.header("📊 Career Insights")
        st.write("Explore different career paths and their characteristics.")
        
        if st.checkbox("Show Career Options"):
            if 'Predicted_Career_Field' in data.columns:
                st.dataframe(data['Predicted_Career_Field'].value_counts().reset_index().rename(
                    columns={'index': 'Career', 'Predicted_Career_Field': 'Frequency'}))
            else:
                st.warning("Career field data not available.")
        
        st.subheader("Popular Career Paths")
        if 'Predicted_Career_Field' in data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            data['Predicted_Career_Field'].value_counts().head(15).plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title("Most Common Career Paths")
            ax.set_xlabel("Frequency")
            st.pyplot(fig)
        else:
            st.warning("No career path data available.")
        
        st.subheader("Career Characteristics")
        if 'Predicted_Career_Field' in data.columns:
            selected_career = st.selectbox(
                "Select a career to learn more:",
                sorted(data['Predicted_Career_Field'].unique()),
                key="career_select"
            )
            
            career_data = data[data['Predicted_Career_Field'] == selected_career]
            
            if not career_data.empty:
                st.write(f"**Typical profile for {selected_career}:**")
                cols = st.columns(3)
                with cols[0]:
                    if 'GPA' in career_data.columns and not career_data['GPA'].isnull().all():
                        st.metric("Average GPA", f"{career_data['GPA'].mean():.1f}")
                    else:
                        st.metric("Average GPA", "N/A")
                with cols[1]:
                    if 'Years_of_Experience' in career_data.columns:
                        st.metric("Avg. Experience", f"{career_data['Years_of_Experience'].mean():.1f} years")
                    else:
                        st.metric("Avg. Experience", "N/A")
                with cols[2]:
                    if 'Interest' in career_data.columns:
                        st.metric("Common Interest", career_data['Interest'].mode()[0])
                    else:
                        st.metric("Common Interest", "N/A")

                st.write("\n**Common characteristics:**")
                if 'Work_Style' in career_data.columns:
                    st.write(f"- Work Style: {career_data['Work_Style'].mode()[0]}")
                if 'Strengths' in career_data.columns:
                    st.write(f"- Strengths: {career_data['Strengths'].mode()[0]}")
                if 'Communication_Skills' in career_data.columns:
                    st.write(f"- Communication: {career_data['Communication_Skills'].mode()[0]}")
            else:
                st.warning("No data available for this career.")
        else:
            st.warning("No career data available for analysis.")

if __name__ == "__main__":
    main()
