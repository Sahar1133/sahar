import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

# -----------------------------
# Set page config - MUST BE FIRST STREAMLIT COMMAND
# -----------------------------
st.set_page_config(
    page_title="Career Path Predictor",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS Styling with Background Image
# -----------------------------
st.markdown("""
<style>
    /* Main container with background image */
    .stApp {
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    
    /* Welcome screen styling */
    .welcome-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .welcome-title {
        font-size: 4rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: 1.5rem;
        margin-bottom: 3rem;
        max-width: 800px;
    }
    
    .start-button {
        background-color: #3498db !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 15px 40px !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    .start-button:hover {
        background-color: #2980b9 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.2) !important;
    }
    
    /* Main content styling */
    .main-container {
        background-color: rgba(255, 255, 255, 0.93);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Header styling */
    .header {
        color: #2c3e50;
        text-align: center;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* Title styling */
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Radio button styling */
    .stRadio>div {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #3498db, #2c3e50);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    
    /* Feature importance table */
    .feature-table {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        padding: 1rem;
        border: 1px solid #e9ecef;
    }
    
    /* Section headers */
    .section-header {
        color: #2c3e50;
        border-bottom: 1px solid #ecf0f1;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
    }
    
    /* Form styling */
    .stForm {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Number input styling */
    .stNumberInput>div>div>input {
        border-radius: 8px !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Select box styling */
    .stSelectbox>div>div>select {
        border-radius: 8px !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants and Mappings
# -----------------------------
LEVEL_MAPPING = {"Low": 0, "Medium": 1, "High": 2}
YES_NO_MAPPING = {"Yes": 1, "No": 0}

# Field of study options
FIELD_OF_STUDY_OPTIONS = [
    "Accounting", "Computer Science", "Medicine", "Law", "Fine Arts", 
    "Education", "Engineering", "Business Administration", "Psychology", 
    "Biology", "Sociology", "Software Engineering", "Physics", "Nursing", 
    "Civil Engineering", "Sports Science", "Journalism", "Architecture", "Chemistry"
]

# Degree options
DEGREE_OPTIONS = ["Diploma", "Bachelors", "Masters", "PhD"]

# Work schedule options
WORK_SCHEDULE_OPTIONS = ["9-5", "Freelance", "Shifts"]

# Location options
LOCATION_OPTIONS = ["Urban", "Rural", "Flexible"]

# Industry options
INDUSTRY_OPTIONS = [
    "Arts", "Construction", "Education", "Finance", 
    "Healthcare", "Law", "Retail", "Tech"
]

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Book1.xlsx", sheet_name="Sheet1")
    return df

# -----------------------------
# Enhanced Preprocessing
# -----------------------------
def preprocess_data(df):
    df = df.copy()
    le_dict = {}
    category_mapping = {}
    
    # First pass: Identify all possible categories for each column
    for col in df.columns:
        if df[col].dtype == 'object' and col != "Predicted_Career_Field":
            # Convert to string in case there are mixed types
            unique_values = df[col].astype(str).unique()
            category_mapping[col] = list(unique_values)
    
    # Second pass: Create label encoders with all known categories
    for col in df.columns:
        if df[col].dtype == 'object' and col != "Predicted_Career_Field":
            le = LabelEncoder()
            # Ensure we're working with strings and handle NaN/None values
            categories = [str(x) for x in category_mapping[col]]
            le.fit(categories)
            df[col] = le.transform(df[col].astype(str))
            le_dict[col] = le
    
    target_le = LabelEncoder()
    df["Predicted_Career_Field"] = target_le.fit_transform(df["Predicted_Career_Field"].astype(str))
    
    return df, le_dict, target_le, category_mapping

# -----------------------------
# Train Model with Feature Selection
# -----------------------------
def train_model(X, y, n_features=10):
    # First train to get feature importances
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    # Select top N features
    selector = SelectFromModel(clf, max_features=n_features, threshold=-np.inf)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    
    # Retrain with selected features
    X_reduced = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )
    clf.fit(X_train, y_train)
    
    return clf, selected_features

# -----------------------------
# Questions dictionary
# -----------------------------
# -----------------------------
# Questions dictionary
# -----------------------------
questions_dict = {
    "Interest": [
        {
            "question": "What is your area of interest?",
            "options": {
                "Arts": "Arts",
                "Business": "Business",
                "Consulting": "Consulting",
                "Design": "Design",
                "Sports": "Sports",
                "Doctor": "Doctor",
                "Education": "Education",
                "Engineering": "Engineering",
                "Finance": "Finance",
                "Hospitality": "Hospitality",
                "Human Resources": "Human Resources",
                "Journalism": "Journalism",
                "Lawyer": "Lawyer",
                "Legal": "Legal",
                "Marketing": "Marketing",
                "Mathematics": "Mathematics",
                "Medical": "Medical",
                "Public Relations": "Public Relations",
                "Sales": "Sales",
                "Teaching": "Teaching",
                "Technology": "Technology"
            }
        }
    ],
    "Work_Style": [
        {
            "question": "How do you prefer to structure your day?",
            "options": {
                "I create and follow my own plan": "Independent",
                "I adjust based on how the day unfolds": "Flexible",
                "I coordinate closely with others": "Collaborative"
            }
        },
        {
            "question": "When working on a group project, what role do you naturally take?",
            "options": {
                "I prefer to work on my own tasks solo": "Independent",
                "I switch roles based on what's needed": "Flexible",
                "I bring people together and coordinate efforts": "Collaborative"
            }
        },
        {
            "question": "How do you feel when plans change last minute?",
            "options": {
                "I find it frustrating and disruptive": "Independent",
                "I take it as a challenge and adapt": "Flexible",
                "I check in with the group and decide together": "Collaborative"
            }
        },
        {
            "question": "What kind of work environment brings out your best?",
            "options": {
                "A quiet space where I work alone with clear goals": "Independent",
                "A dynamic space where I can change pace or tasks": "Flexible",
                "A lively team setting with frequent interactions": "Collaborative"
            }
        },
        {
            "question": "How do you prefer to receive assignments?",
            "options": {
                "I like full autonomy to get things done my way": "Independent",
                "I'm okay with general goals and figuring it out": "Flexible",
                "I prefer discussing with the team and aligning tasks": "Collaborative"
            }
        }
    ],
    "Strengths": [
        {
            "question": "When you're faced with a complex issue, what's your first instinct?",
            "options": {
                "Create something new and imaginative": "Creative",
                "Strategize and plan the steps carefully": "Strategic",
                "Break it down into logical, solvable parts": "Analytical"
            }
        },
        {
            "question": "Which compliment resonates the most with you?",
            "options": {
                "You're incredibly imaginative!": "Creative",
                "You always see the big picture.": "Strategic",
                "You have excellent problem-solving skills.": "Analytical"
            }
        },
        {
            "question": "Your friends say you excel at:",
            "options": {
                "Coming up with original ideas": "Creative",
                "Making well-thought-out plans": "Strategic",
                "Solving tricky logic puzzles or analyzing data": "Analytical"
            }
        },
        {
            "question": "In a team setting, you're usually the one who:",
            "options": {
                "Brainstorms fresh, out-of-the-box solutions": "Creative",
                "Organizes the project timeline and milestones": "Strategic",
                "Checks for inconsistencies and fine-tunes details": "Analytical"
            }
        },
        {
            "question": "Which task sounds most fun to you?",
            "options": {
                "Designing a logo or writing a story": "Creative",
                "Mapping out a 6-month growth strategy": "Strategic",
                "Analyzing why something failed and how to fix it": "Analytical"
            }
        }
    ],
    "Communication_Skills": [
        {
            "question": "During group discussions, how do you usually participate?",
            "options": {
                "I lead the conversation and clarify ideas": "High",
                "I share my thoughts and support others": "Medium",
                "I prefer to stay quiet and observe": "Low"
            }
        },
        {
            "question": "When asked to present an idea to a group, how do you feel?",
            "options": {
                "Confident and excited to explain": "High",
                "Slightly nervous, but I can do it well": "Medium",
                "I need preparation to feel okay": "Low"
            }
        },
        {
            "question": "If a misunderstanding occurs in a team, what's your response?",
            "options": {
                "I take initiative to resolve it quickly and clearly": "High",
                "I help clarify things if I'm asked": "Medium",
                "I listen and let others take the lead": "Low"
            }
        },
        {
            "question": "How comfortable are you speaking with people from different backgrounds?",
            "options": {
                "Very comfortable - I adapt my style easily": "High",
                "Fairly comfortable in most situations": "Medium",
                "I'm a bit hesitant but try to manage": "Low"
            }
        },
        {
            "question": "What do others most often say about your communication?",
            "options": {
                "Clear, engaging, and persuasive": "High",
                "Polite, helpful, and articulate": "Medium",
                "Reserved, but gets the point across": "Low"
            }
        }
    ],
    "Leadership_Skills": [
        {
            "question": "When a team faces a challenge, what is your typical role?",
            "options": {
                "I take charge and guide the team toward a solution": "High",
                "I support the leader and help implement ideas": "Medium",
                "I wait for others to take charge and follow along": "Low"
            }
        },
        {
            "question": "How do you feel about making important decisions for the team?",
            "options": {
                "I feel confident and ready to make decisions": "High",
                "I consult others and make decisions together": "Medium",
                "I prefer to avoid decision-making whenever possible": "Low"
            }
        },
        {
            "question": "How do you motivate your team during tough times?",
            "options": {
                "I inspire and encourage the team to keep pushing forward": "High",
                "I offer support and try to keep the team focused": "Medium",
                "I struggle to motivate others and tend to withdraw": "Low"
            }
        },
        {
            "question": "When it comes to delegating tasks, how do you approach it?",
            "options": {
                "I delegate effectively and ensure everyone has a clear role": "High",
                "I delegate when needed but prefer to take on tasks myself": "Medium",
                "I have difficulty delegating tasks and often end up doing everything": "Low"
            }
        },
        {
            "question": "How do you handle conflicts within the team?",
            "options": {
                "I address conflicts directly and try to resolve them constructively": "High",
                "I intervene when necessary, but prefer not to get too involved": "Medium",
                "I avoid conflicts and let others handle them": "Low"
            }
        }
    ],
    "Teamwork_Skills": [
        {
            "question": "How do you contribute to a team's success?",
            "options": {
                "I actively collaborate, offering ideas and support to everyone": "High",
                "I participate when needed and contribute in specific areas": "Medium",
                "I mostly work alone and only contribute when asked": "Low"
            }
        },
        {
            "question": "When a team project is struggling, what is your response?",
            "options": {
                "I work closely with the team to identify solutions and keep the momentum going": "High",
                "I try to help but often wait for someone else to lead the way": "Medium",
                "I tend to focus on my own tasks and avoid getting involved in team challenges": "Low"
            }
        },
        {
            "question": "How comfortable are you with giving and receiving feedback in a team?",
            "options": {
                "I welcome feedback and offer constructive criticism to others": "High",
                "I accept feedback well, but I'm cautious when providing it": "Medium",
                "I find it difficult to give feedback and feel uncomfortable receiving it": "Low"
            }
        },
        {
            "question": "When a team decision is made, how do you react if you disagree?",
            "options": {
                "I express my concerns respectfully and help the team reconsider if needed": "High",
                "I share my opinion, but generally go along with the group's decision": "Medium",
                "I stay quiet and follow the decision without voicing my disagreement": "Low"
            }
        },
        {
            "question": "How do you handle situations where team members aren't pulling their weight?",
            "options": {
                "I address the issue directly, offering help and guidance to ensure the team succeeds": "High",
                "I express my concerns, but I usually wait for someone else to step in": "Medium",
                "I avoid addressing the issue and focus on my own work": "Low"
            }
        }
    ],
    "Decision_Making": [
        {
            "question": "When faced with a difficult decision, how do you approach it?",
            "options": {
                "I gather information, weigh the pros and cons, and make an informed decision": "High",
                "I seek input from others and try to make a balanced decision": "Medium",
                "I avoid making decisions and leave them to others": "Low"
            }
        },
        {
            "question": "How do you feel when you have to make decisions under pressure?",
            "options": {
                "I remain calm and focused, making decisions efficiently": "High",
                "I feel a bit stressed but manage to make decisions with some effort": "Medium",
                "I struggle to make decisions quickly and may freeze under pressure": "Low"
            }
        },
        {
            "question": "How do you handle situations where there's no clear right or wrong choice?",
            "options": {
                "I analyze the situation, trust my instincts, and choose the best option": "High",
                "I consider the potential outcomes and pick the best compromise": "Medium",
                "I struggle to decide and often leave the choice up to others": "Low"
            }
        },
        {
            "question": "When making decisions, how often do you consider the long-term consequences?",
            "options": {
                "I always consider the long-term impact and how it affects everyone involved": "High",
                "I try to think about the long-term, but sometimes focus on short-term needs": "Medium",
                "I mostly focus on immediate outcomes and don't think much about the future": "Low"
            }
        },
        {
            "question": "How do you react if a decision you made turns out to be wrong?",
            "options": {
                "I take responsibility, learn from it, and adjust my approach in the future": "High",
                "I acknowledge the mistake and try to correct it where possible": "Medium",
                "I avoid taking responsibility and tend to blame others for the mistake": "Low"
            }
        }
    ],
 "Aptitude_Level": [
        {
            "question": "If 5 machines take 5 minutes to make 5 products, how long will 100 machines take to make 100 products?",
            "options": {
                "5 minutes": "High",
                "50 minutes": "Medium",
                "100 minutes": "Low"
            }
        },
        {
            "question": "Find the missing number in the sequence: 2, 6, 12, 20, __, 42",
            "options": {
                "28": "High",
                "30": "Medium",
                "32": "Low"
            }
        },
        {
            "question": "A bat and a ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
            "options": {
                " $0.05": "High",
                "0.10": "Medium",
                "1.00": "Low"
            }
        },
        {
            "question": "If all Bloops are Razzies, and all Razzies are Lazzies, then are all Bloops definitely Lazzies?",
            "options": {
                "Yes": "High",
                "Maybe": "Medium",
                "No": "Low"
            }
        },
        {
            "question": "Which shape comes next in the pattern? [🔺🔺🔷🔺🔺🔷🔺🔺❓]",
            "options": {
                "🔷": "High",
                "🔺": "Medium",
                "⭕": "Low"
            }
        }
    ],

    "Adaptability": [
        {
            "question": "How do you react when plans change at the last minute?",
            "options": {
                "I quickly adjust and move forward": "High",
                "I feel a bit uneasy but manage": "Medium",
                "I get frustrated and find it hard to cope": "Low"
            }
        },
        {
            "question": "You're assigned to work with someone whose methods are very different from yours. What do you do?",
            "options": {
                "Try to understand and collaborate": "High",
                "Keep to your style but tolerate theirs": "Medium",
                "Struggle to work effectively with them": "Low"
            }
        },
        {
            "question": "How do you feel about learning new tools or technologies?",
            "options": {
                "Excited and eager": "High",
                "Neutral - depends on the need": "Medium",
                "Hesitant or avoid it unless required": "Low"
            }
        },
        {
            "question": "What's your response to unexpected challenges at work or school?",
            "options": {
                "I stay calm and look for solutions": "High",
                "I take time to process but respond": "Medium",
                "I often feel overwhelmed": "Low"
            }
        },
        {
            "question": "If you're moved to a new team or environment, how do you handle it?",
            "options": {
                "I enjoy meeting new people and learning": "High",
                "I take time to settle but adapt eventually": "Medium",
                "I prefer not to change teams at all": "Low"
            }
        }
    ],
    "Time_Management": [
        {
            "question": "How often do you create a to-do list or plan your day in advance?",
            "options": {
                "Daily - I rely on it to stay organized": "High",
                "Occasionally, when I have a lot to do": "Medium",
                "Rarely - I go with the flow": "Low"
            }
        },
        {
            "question": "You have a deadline in 3 days. When do you typically start working on it?",
            "options": {
                "Immediately - I like to stay ahead": "High",
                "A day before - I work better under pressure": "Medium",
                "Last minute or after the deadline - I struggle to start": "Low"
            }
        },
        {
            "question": "You have multiple tasks to complete. What do you do first?",
            "options": {
                "Prioritize based on urgency and importance": "High",
                "Start with whichever feels easiest": "Medium",
                "Delay and often jump between tasks": "Low"
            }
        },
        {
            "question": "How often do distractions (e.g., social media, chatting) affect your productivity?",
            "options": {
                "Rarely - I know how to stay focused": "High",
                "Sometimes - I try to control it": "Medium",
                "Frequently - I lose a lot of time": "Low"
            }
        },
        {
            "question": "If you're running out of time on a task, you...",
            "options": {
                "Stay calm, re-prioritize, and focus": "High",
                "Rush through and hope for the best": "Medium",
                "Panic and sometimes give up or skip parts": "Low"
            }
        }
    ],
    "Problem_Solving": [
        {
            "question": "When faced with a complex problem, what is your first reaction?",
            "options": {
                "Break it down and analyze step by step": "High",
                "Try a few things and see what works": "Medium",
                "Feel overwhelmed and unsure how to begin": "Low"
            }
        },
        {
            "question": "How do you handle situations when your first solution doesn't work?",
            "options": {
                "I look for alternative solutions immediately": "High",
                "I try the same method a few times": "Medium",
                "I get stuck and find it hard to continue": "Low"
            }
        },
        {
            "question": "You're under pressure to solve an urgent issue. What do you do?",
            "options": {
                "Stay calm and logically evaluate the best course": "High",
                "Act quickly, even without full analysis": "Medium",
                "Struggle to focus and feel stressed": "Low"
            }
        },
        {
            "question": "How do you approach unfamiliar problems?",
            "options": {
                "I enjoy the challenge and use reasoning": "High",
                "I try what has worked in similar cases": "Medium",
                "I usually wait for guidance or help": "Low"
            }
        },
        {
            "question": "You encounter a recurring issue in your work. What's your approach?",
            "options": {
                "Identify the root cause and find a permanent fix": "High",
                "Apply temporary solutions to keep moving": "Medium",
                "Avoid dealing with it until it becomes critical": "Low"
            }
        }
    ],
    "Emotional_Intelligence": [
        {
            "question": "How do you react when you feel angry or frustrated?",
            "options": {
                "I recognize my emotions and take a step back to calm down": "High",
                "I try to ignore it or bottle it up": "Medium",
                "I express it right away, often without thinking": "Low"
            }
        },
        {
            "question": "When a team member is upset, what is your usual response?",
            "options": {
                "I try to understand their feelings and offer support": "High",
                "I listen but don't always know how to help": "Medium",
                "I avoid them until they calm down": "Low"
            }
        },
        {
            "question": "How often do you reflect on your own emotional reactions in situations?",
            "options": {
                "Frequently - I think about how I felt and why": "High",
                "Occasionally - I sometimes analyze my emotions": "Medium",
                "Rarely - I don't really reflect on how I feel": "Low"
            }
        },
        {
            "question": "If someone criticizes your work, how do you typically respond?",
            "options": {
                "I listen calmly, process the feedback, and improve": "High",
                "I feel defensive but try to improve after some thought": "Medium",
                "I get upset and have difficulty accepting the criticism": "Low"
            }
        },
        {
            "question": "How do you manage your emotions in stressful situations?",
            "options": {
                "I stay composed and focus on finding a solution": "High",
                "I feel stressed but try to push through": "Medium",
                "I get overwhelmed and may struggle to focus": "Low"
            }
        }
    ],
    "Stress_Tolerance": [
        {
            "question": "How do you typically feel when you are faced with multiple deadlines or tasks at once?",
            "options": {
                "I stay calm, prioritize, and work through the tasks methodically": "High",
                "I feel stressed but manage to get through it with some effort": "Medium",
                "I feel overwhelmed and struggle to complete tasks": "Low"
            }
        },
        {
            "question": "When you face a challenging or unexpected situation, how do you react?",
            "options": {
                "I stay composed, think logically, and find a solution": "High",
                "I get anxious but manage to pull through eventually": "Medium",
                "I panic and have difficulty managing the situation": "Low"
            }
        },
        {
            "question": "When dealing with stressful situations, how easily do you recover emotionally?",
            "options": {
                "Quickly - I manage stress well and bounce back fast": "High",
                "It takes some time, but I eventually regain my composure": "Medium",
                "I find it difficult to recover and often carry stress with me": "Low"
            }
        },
        {
            "question": "In high-pressure environments, how do you perform?",
            "options": {
                "I perform well, even under pressure, and stay productive": "High",
                "I feel pressure but still manage to meet expectations": "Medium",
                "I struggle to perform under pressure and may miss deadlines": "Low"
            }
        },
        {
            "question": "How do you feel when your plans or routines are disrupted unexpectedly?",
            "options": {
                "I adjust quickly without much stress": "High",
                "I feel unsettled but manage to adapt over time": "Medium",
                "I get frustrated and have difficulty adjusting": "Low"
            }
        }
    ],
    "Learning_Style": [
        {
            "question": "When you are studying or learning new information, which method works best for you?",
            "options": {
                "Writing notes, summarizing what I've learned, or reading books/articles": "Reading/Writing",
                "Listening to podcasts, lectures, or discussions": "Auditory",
                "Watching videos, diagrams, or other visual content": "Visual",
                "Actively doing hands-on activities, practicing tasks, or using physical examples": "Kinesthetic"
            }
        },
        {
            "question": "How do you prefer to receive instructions for a new task or project?",
            "options": {
                "I prefer written instructions or guides": "Reading/Writing",
                "I like to hear someone explain the steps aloud": "Auditory",
                "I prefer to see a demonstration or visual guide first": "Visual",
                "I prefer to try things myself and learn through doing": "Kinesthetic"
            }
        },
        {
            "question": "When remembering information for an exam or presentation, what works best for you?",
            "options": {
                "Reading and writing things down repeatedly": "Reading/Writing",
                "Listening to recorded lectures or talking it out with others": "Auditory",
                "Visualizing concepts or drawing diagrams to recall information": "Visual",
                "Acting out scenarios or using hands-on experience to remember": "Kinesthetic"
            }
        },
        {
            "question": "How do you prefer to take notes during a class or meeting?",
            "options": {
                "I take detailed notes and often write everything down": "Reading/Writing",
                "I prefer listening and taking short notes or summarizing key points": "Auditory",
                "I use diagrams, pictures, or colors to organize my notes": "Visual",
                "I like to take notes by doing or using physical objects to represent ideas": "Kinesthetic"
            }
        },
        {
            "question": "When faced with a new topic, how do you prefer to start learning?",
            "options": {
                "Reading books, articles, or written material about the topic": "Reading/Writing",
                "Listening to discussions, podcasts, or lectures": "Auditory",
                "Watching videos, charts, or any visual material that illustrates the topic": "Visual",
                "Engaging in hands-on practice, workshops, or interactive activities": "Kinesthetic"
            }
        }
    ],
    "Risk_Tolerance": [
        {
            "question": "When presented with a new opportunity that has both high rewards and high risk, what is your initial reaction?",
            "options": {
                "I feel excited and consider how to take it strategically": "High",
                "I weigh the pros and cons carefully before deciding": "Medium",
                "I avoid it unless the risk is minimal": "Low"
            }
        },
        {
            "question": "How do you usually make important decisions with uncertain outcomes?",
            "options": {
                "I'm comfortable deciding even if all details aren't clear": "High",
                "I take time to analyze but eventually take a moderate stance": "Medium",
                "I prefer clear and safe outcomes before acting": "Low"
            }
        },
        {
            "question": "In group projects, are you willing to try unconventional or bold approaches?",
            "options": {
                "Yes, I often suggest bold or innovative methods": "High",
                "Sometimes, if the situation really needs it": "Medium",
                "I prefer sticking to tried-and-tested approaches": "Low"
            }
        },
        {
            "question": "If you had to choose between a stable job and a startup with uncertain income but high potential, what would you pick?",
            "options": {
                "The startup - I'm willing to take the chance": "High",
                "Depends - I would think deeply before choosing": "Medium",
                "The stable job - security is more important": "Low"
            }
        },
        {
            "question": "When taking on a new challenge or role with unclear expectations, how do you respond?",
            "options": {
                "I embrace it and learn along the way": "High",
                "I feel nervous but accept it after preparation": "Medium",
                "I feel uncomfortable and prefer clarity before starting": "Low"
            }
        }
    ],
    "Introvert_Extrovert_Score": [
        {
            "question": "How do you usually feel after spending a few hours at a lively social gathering?",
            "options": {
                "Energized and excited to keep socializing": "High",
                "It was fun, but I need a bit of alone time now": "Medium",
                "Drained and ready for quiet solitude": "Low"
            }
        },
        {
            "question": "When working on a project, what environment do you prefer?",
            "options": {
                "A team setting where I can discuss and share ideas actively": "High",
                "A mix of solo work and occasional collaboration": "Medium",
                "A quiet space where I can focus alone": "Low"
            }
        },
        {
            "question": "How do you usually respond in group discussions or meetings?",
            "options": {
                "I often lead, talk freely, and enjoy being heard": "High",
                "I contribute when I have something important to say": "Medium",
                "I mostly observe and prefer listening over speaking": "Low"
            }
        },
        {
            "question": "What's your idea of an ideal weekend?",
            "options": {
                "Going out with friends, meeting new people, attending events": "High",
                "A balance of going out and staying in": "Medium",
                "Reading a book, watching movies, or relaxing alone": "Low"
            }
        },
        {
            "question": "How do you handle networking events or public speaking?",
            "options": {
                "I enjoy them and look forward to the interaction": "High",
                "I do okay, depending on the mood or setting": "Medium",
                "I feel nervous or avoid them if possible": "Low"
            }
        }
    ],
    "Workplace_Preference": [
        {
            "question": "What type of setting makes you feel most productive during the day?",
            "options": {
                "A quiet home office": "Remote",
                "A collaborative open office": "On-site",
                "A mix depending on the task": "Hybrid"
            }
        },
        {
            "question": "If you had an important task, where would you prefer to work?",
            "options": {
                "Home": "Remote",
                "Company Office": "On-site",
                "Library or Café": "Hybrid"
            }
        },
        {
            "question": "How do you like to interact with teammates?",
            "options": {
                "Messaging tools": "Remote",
                "Face-to-face": "On-site",
                "Mix of both": "Hybrid"
            }
        },
        {
            "question": "Do you enjoy commuting or starting your day directly?",
            "options": {
                "Hate commuting": "Remote",
                "Enjoy the routine": "On-site",
                "Sometimes like it, sometimes don't": "Hybrid"
            }
        },
        {
            "question": "How important are spontaneous conversations at work?",
            "options": {
                "Not important": "Remote",
                "Very important": "On-site",
                "Somewhat important": "Hybrid"
            }
        }
    ],
    "Work_Life_Balance_Preference": [
        {
            "question": "What do you typically do after work?",
            "options": {
                "Unplug and relax": "High",
                "Catch up on more tasks": "Low",
                "Check emails but relax later": "Moderate"
            }
        },
        {
            "question": "How do you react to late work requests?",
            "options": {
                "Try to defer or say no": "High",
                "Accept as part of the job": "Low",
                "Depends on urgency": "Moderate"
            }
        },
        {
            "question": "Describe your ideal workday.",
            "options": {
                "9-5 with evenings free": "High",
                "Extended hours for growth": "Low",
                "Balanced hours with breaks": "Moderate"
            }
        },
        {
            "question": "What do you value more?",
            "options": {
                "Health and energy": "High",
                "Results and goals": "Low",
                "A mix of both": "Moderate"
            }
        },
        {
            "question": "How often do you check emails after hours?",
            "options": {
                "Never": "High",
                "Frequently": "Low",
                "Occasionally": "Moderate"
            }
        }
    ],
    "Salary_Expectation": [
        {
            "question": "What makes a job offer attractive?",
            "options": {
                "Salary": "High",
                "Culture and learning": "Low",
                "A mix": "Moderate"
            }
        },
        {
            "question": "Two jobs are equally interesting. Which do you choose?",
            "options": {
                "Higher-paying one": "High",
                "One with more growth": "Low",
                "Balanced between both": "Moderate"
            }
        },
        {
            "question": "How do you feel if a peer earns more?",
            "options": {
                "Unfair, I deserve that too": "High",
                "Doesn't matter": "Low",
                "Slightly envious but okay": "Moderate"
            }
        },
        {
            "question": "What's more satisfying?",
            "options": {
                "Financial reward": "High",
                "Recognition or freedom": "Low",
                "All of the above": "Moderate"
            }
        },
        {
            "question": "What do you research first in a job post?",
            "options": {
                "Salary details": "High",
                "Role expectations": "Low",
                "Company background": "Moderate"
            }
        }
    ],
    "Value_Priority": [
        {
            "question": "What matters most in your career?",
            "options": {
                "Security and income": "Stability",
                "Creating change": "Purpose",
                "Inventing new solutions": "Innovation"
            }
        },
        {
            "question": "Would you rather:",
            "options": {
                "Improve an existing system": "Stability",
                "Start something new": "Innovation",
                "Solve a social issue": "Purpose"
            }
        },
        {
            "question": "What excites you more?",
            "options": {
                "Order and structure": "Stability",
                "Risk and novelty": "Innovation",
                "Impact and meaning": "Purpose"
            }
        },
        {
            "question": "What's a fulfilling success story?",
            "options": {
                "Financial independence": "Stability",
                "Created change in society": "Purpose",
                "Built a new product or tool": "Innovation"
            }
        },
        {
            "question": "Solve one problem in your field:",
            "options": {
                "Resource inefficiency": "Stability",
                "Lack of access": "Purpose",
                "Technical limitations": "Innovation"
            }
        }
    ],
    "Openness": [
        {
            "question": "What's your reaction to new, abstract ideas?",
            "options": {
                "Excited and curious": "High",
                "Interested but cautious": "Medium",
                "Skeptical or uninterested": "Low"
            }
        },
        {
            "question": "Would you enjoy a spontaneous trip to a place you've never been?",
            "options": {
                "Absolutely!": "High",
                "Maybe, with some planning": "Medium",
                "I'd rather not": "Low"
            }
        },
        {
            "question": "How do you feel about trying out-of-the-box solutions?",
            "options": {
                "I encourage them": "High",
                "I consider them but prefer proven ways": "Medium",
                "I avoid untested methods": "Low"
            }
        },
        {
            "question": "Do you often engage in creative hobbies?",
            "options": {
                "Yes, regularly": "High",
                "Occasionally": "Medium",
                "Rarely or never": "Low"
            }
        },
        {
            "question": "If someone offered a job in a totally new domain, you'd:",
            "options": {
                "Explore it!": "High",
                "Consider cautiously": "Medium",
                "Decline due to risk": "Low"
            }
        }
    ],
    "Conscientiousness": [
        {
            "question": "How do you usually manage your tasks?",
            "options": {
                "I plan everything ahead": "High",
                "I manage as things come": "Medium",
                "I often forget or delay": "Low"
            }
        },
        {
            "question": "Do you stick to deadlines?",
            "options": {
                "Always": "High",
                "Most of the time": "Medium",
                "Not really": "Low"
            }
        },
        {
            "question": "Your workspace is usually:",
            "options": {
                "Neat and organized": "High",
                "Manageably messy": "Medium",
                "Chaotic": "Low"
            }
        },
        {
            "question": "When given a responsibility, you:",
            "options": {
                "Follow through completely": "High",
                "Try your best": "Medium",
                "Struggle to finish": "Low"
            }
        },
        {
            "question": "What's your approach to goal setting?",
            "options": {
                "Break into steps and track": "High",
                "Set goals, review occasionally": "Medium",
                "Go with the flow": "Low"
            }
        }
    ],
    "Extraversion": [
        {
            "question": "What gives you energy?",
            "options": {
                "Social interaction": "High",
                "A mix of social and alone time": "Medium",
                "Solitude": "Low"
            }
        },
        {
            "question": "In a room full of strangers, you:",
            "options": {
                "Mingle quickly": "High",
                "Talk to a few": "Medium",
                "Stay reserved": "Low"
            }
        },
        {
            "question": "Weekend plans look like:",
            "options": {
                "Parties or outings": "High",
                "Social + personal time": "Medium",
                "Relaxing alone": "Low"
            }
        },
        {
            "question": "Do you prefer working in teams or alone?",
            "options": {
                "Teams": "High",
                "Depends on the task": "Medium",
                "Alone": "Low"
            }
        },
        {
            "question": "How often do you speak first in group chats?",
            "options": {
                "Frequently": "High",
                "Occasionally": "Medium",
                "Rarely": "Low"
            }
        }
    ],
    "Agreeableness": [
        {
            "question": "When a team member makes a mistake, how do you respond?",
            "options": {
                "Help them correct it gently": "High",
                "Point it out constructively": "Medium",
                "Get frustrated or blame": "Low"
            }
        },
        {
            "question": "How often do you compromise during disagreements?",
            "options": {
                "Most of the time": "High",
                "When it makes sense": "Medium",
                "Rarely - I stand my ground": "Low"
            }
        },
        {
            "question": "What do others often say about you?",
            "options": {
                "Kind and easy to work with": "High",
                "Honest and balanced": "Medium",
                "Straightforward, even if harsh": "Low"
            }
        },
        {
            "question": "If someone takes credit for your work, you:",
            "options": {
                "Let it go to keep peace": "High",
                "Clarify things politely": "Medium",
                "Directly confront them": "Low"
            }
        },
        {
            "question": "You're asked to help with a project last minute. You:",
            "options": {
                "Agree happily": "High",
                "Help if you can manage": "Medium",
                "Refuse if it's inconvenient": "Low"
            }
        }
    ],
    "Neuroticism": [
        {
            "question": "How do you handle criticism?",
            "options": {
                "Stay calm and reflect": "Low",
                "Feel a bit affected but move on": "Medium",
                "Take it personally and dwell on it": "High"
            }
        },
        {
            "question": "How often do you feel anxious about tasks?",
            "options": {
                "Rarely": "Low",
                "Occasionally": "Medium",
                "Frequently": "High"
            }
        },
        {
            "question": "When things don't go as planned, your reaction is:",
            "options": {
                "Adjust and stay positive": "Low",
                "Feel a little stressed": "Medium",
                "Get overwhelmed": "High"
            }
        },
        {
            "question": "How stable are your moods throughout the week?",
            "options": {
                "Very stable": "Low",
                "Sometimes fluctuate": "Medium",
                "Often up and down": "High"
            }
        },
        {
            "question": "During stressful situations, you:",
            "options": {
                "Stay composed": "Low",
                "Need time to process": "Medium",
                "Feel like breaking down": "High"
            }
        }
    ],

    "Technical_Skill_Level": [
        {
            "question": "When using a new software or tool, how do you usually proceed?",
            "options": {
                "I explore it confidently and figure it out on my own": "High",
                "I need some guidance or tutorials to get started": "Medium",
                "I feel unsure and prefer someone else sets it up": "Low"
            }
        },
        {
            "question": "How comfortable are you with solving technical issues like setting up a network, debugging code, or configuring systems?",
            "options": {
                "Very comfortable - I enjoy tackling such problems": "High",
                "Somewhat comfortable - I can manage with help or resources": "Medium",
                "Not comfortable - I usually need someone else to fix it": "Low"
            }
        },
        {
            "question": "When given a task that involves coding, data handling, or technical tools, how do you respond?",
            "options": {
                "I'm confident and often take the lead on such tasks": "High",
                "I can contribute but may need assistance": "Medium",
                "I try to avoid it or rely heavily on others": "Low"
            }
        },
        {
            "question": "How often do you update or improve your technical knowledge or skills (e.g., learning new tools, coding languages, etc.)?",
            "options": {
                "Frequently - I enjoy keeping up with new technologies": "High",
                "Occasionally - when needed for a task or job": "Medium",
                "Rarely - I stick with what I already know": "Low"
            }
        },
        {
            "question": "Have you ever built or contributed to a tech-based project (app, website, automation, etc.)?",
            "options": {
                "Yes, multiple times - I actively build or contribute to such projects": "High",
                "Once or twice - I've done it but with some help": "Medium",
                "No - I haven't been involved in such work": "Low"
            }
        }
    ],
    "Location_Preference": [
        {
            "question": "Where would you prefer to work?",
            "options": {
                "Urban area with many opportunities": "Urban",
                "Rural area with peaceful environment": "Rural",
                "I'm flexible about location": "Flexible"
            }
        }
    ],
    "Willing_to_Relocate": [
        {
            "question": "Are you willing to relocate for a job opportunity?",
            "options": {
                "Yes, I'm open to relocating": "Yes",
                "No, I prefer to stay in my current location": "No"
            }
        }
    ],
    "Industry_of_Experience": [
        {
            "question": "Which industry do you have the most experience in?",
            "options": {
                "Arts": "Arts",
                "Construction": "Construction",
                "Education": "Education",
                "Finance": "Finance",
                "Healthcare": "Healthcare",
                "Law": "Law",
                "Retail": "Retail",
                "Tech": "Tech"
            }
        }
    ],
    "Internship_Experience": [
        {
            "question": "Do you have any internship experience?",
            "options": {
                "Yes": "Yes",
                "No": "No"
            }
        }
    ],
    "Remote_Work_Experience": [
        {
            "question": "Do you have experience working remotely?",
            "options": {
                "Yes": "Yes",
                "No": "No"
            }
        }
    ],
    "Routine_vs_Dynamic_Work": [
        {
            "question": "How do you feel about doing the same tasks every day at work?",
            "options": {
                "I prefer consistency and knowing what to expect": "Routine",
                "I get bored quickly and prefer variety in my tasks": "Dynamic"
            }
        },
        {
            "question": "When given a new task outside your usual responsibilities, what's your reaction?",
            "options": {
                "I feel uncomfortable and prefer to stay within my known role": "Routine",
                "I feel excited and enjoy taking on new challenges": "Dynamic"
            }
        },
        {
            "question": "Which work environment do you find more satisfying?",
            "options": {
                "A stable, well-structured job with clear daily duties": "Routine",
                "A flexible, fast-paced job where tasks change often": "Dynamic"
            }
        },
        {
            "question": "How do you handle unexpected changes in your workday or plans?",
            "options": {
                "I prefer not to have surprises and stick to the schedule": "Routine",
                "I adjust quickly and often enjoy the change": "Dynamic"
            }
        },
        {
            "question": "Would you rather:",
            "options": {
                "Master one role and perform it efficiently every day": "Routine",
                "Rotate between different roles and learn multiple skills": "Dynamic"
            }
        }
    ],
    "Creativity_Score": [
        {
            "question": "When faced with a problem, how do you usually approach finding a solution?",
            "options": {
                "I look for new, unconventional ways to solve it": "High",
                "I consider some different options but mostly rely on proven methods": "Medium",
                "I prefer sticking to the usual, well-established solutions": "Low"
            }
        },
        {
            "question": "How often do you come up with new ideas or suggestions in work or daily life?",
            "options": {
                "Frequently - I enjoy brainstorming and innovating": "High",
                "Sometimes - when the situation calls for it": "Medium",
                "Rarely - I don't usually think about new ideas": "Low"
            }
        },
        {
            "question": "Do you enjoy activities like drawing, writing, music, or creative hobbies?",
            "options": {
                "Yes, I actively engage in creative activities": "High",
                "Occasionally, I try creative hobbies but not regularly": "Medium",
                "Not really, I prefer more practical or routine tasks": "Low"
            }
        },
        {
            "question": "How comfortable are you experimenting with new approaches at work or school?",
            "options": {
                "Very comfortable - I like trying different methods": "High",
                "Somewhat comfortable - I try new things if necessary": "Medium",
                "Uncomfortable - I prefer to follow standard procedures": "Low"
            }
        },
        {
            "question": "When reading or learning, do you prefer material that encourages imagination or creative thinking?",
            "options": {
                "Yes, I enjoy content that inspires new ideas": "High",
                "Sometimes, but I also like straightforward information": "Medium",
                "No, I prefer factual, clear-cut material": "Low"
            }
        }
    ],
    "LinkedIn_Portfolio": [
        {
            "question": "Do you have a LinkedIn profile or professional portfolio?",
            "options": {
                "Yes": "Yes",
                "No": "No"
            }
        }
    ],
    "Public_Speaking_Experience": [
        {
            "question": "Do you have any public speaking experience?",
            "options": {
                "Yes": "Yes",
                "No": "No"
            }
        }
    ]
}


# -----------------------------
# Ask Questions - Improved Version
# -----------------------------
def ask_questions(features):
    st.subheader("Answer the following questions:")
    user_input = {}
    
    # Initialize session state for selected questions if not exists
    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = {}
    
    # Process all features in order
    for feature in features:
        # Check if feature has questions in dictionary
        if feature in questions_dict and len(questions_dict[feature]) > 0:
            # If we haven't selected a question for this feature yet, pick one randomly
            if feature not in st.session_state.selected_questions:
                st.session_state.selected_questions[feature] = np.random.choice(questions_dict[feature])
            
            # Get the randomly selected question
            qa = st.session_state.selected_questions[feature]
            question = qa["question"]
            options = list(qa["options"].keys())
            
            # Display the question and get response
            response = st.radio(question, options, key=f"q_{feature}")
            level = qa["options"][response]
            
            # Map response to numerical value
            if level in LEVEL_MAPPING:
                user_input[feature] = LEVEL_MAPPING[level]
            elif level in YES_NO_MAPPING:
                user_input[feature] = YES_NO_MAPPING[level]
            else:
                user_input[feature] = level  # Keep as-is for other mappings
                
        else:
            # Special handling for specific fields
            if feature == "GPA":
                user_input[feature] = st.number_input(
                    f"What is your {feature.replace('_', ' ')}?",
                    min_value=0.0, max_value=4.0, value=3.0, step=0.1,
                    key=f"num_{feature}"
                )
            elif feature == "Years_of_Experience":
                user_input[feature] = st.number_input(
                    f"How many years of {feature.replace('_', ' ').lower()} do you have?",
                    min_value=0, max_value=50, value=2, step=1,
                    key=f"num_{feature}"
                )
            elif feature == "Certifications_Count":
                user_input[feature] = st.number_input(
                    f"How many {feature.replace('_', ' ').lower()} do you have?",
                    min_value=0, max_value=100, value=2, step=1,
                    key=f"num_{feature}"
                )
            elif feature == "Field_of_Study":
                user_input[feature] = st.selectbox(
                    "What is your field of study?",
                    options=FIELD_OF_STUDY_OPTIONS,
                    key=f"sel_{feature}"
                )
            elif feature == "Highest_Degree":
                user_input[feature] = st.selectbox(
                    "What is your highest degree?",
                    options=DEGREE_OPTIONS,
                    key=f"sel_{feature}"
                )
            elif feature == "Courses_Completed":
                user_input[feature] = st.number_input(
                    "How many courses have you completed?",
                    min_value=0, max_value=10, value=5, step=1,
                    key=f"num_{feature}"
                )
            elif feature == "Work_Hour_Flexibility":
                user_input[feature] = st.selectbox(
                    "What type of work schedule do you prefer?",
                    options=WORK_SCHEDULE_OPTIONS,
                    key=f"sel_{feature}"
                )
            elif feature == "GitHub_Repos":
                user_input[feature] = st.number_input(
                    "How many GitHub repositories have you created?",
                    min_value=0, max_value=20, value=2, step=1,
                    key=f"num_{feature}"
                )
            elif feature == "Location_Preference":
                user_input[feature] = st.selectbox(
                    "Where would you prefer to work?",
                    options=LOCATION_OPTIONS,
                    key=f"sel_{feature}"
                )
            elif feature in ["Willing_to_Relocate", "Internship_Experience", 
                            "Remote_Work_Experience", "LinkedIn_Portfolio", 
                            "Public_Speaking_Experience"]:
                user_input[feature] = st.selectbox(
                    f"{feature.replace('_', ' ')}?",
                    options=["Yes", "No"],
                    key=f"sel_{feature}"
                )
                user_input[feature] = YES_NO_MAPPING[user_input[feature]]
            elif feature == "Industry_of_Experience":
                user_input[feature] = st.selectbox(
                    "Which industry do you have the most experience in?",
                    options=INDUSTRY_OPTIONS,
                    key=f"sel_{feature}"
                )
            else:
                # For other features that don't have questions in the dict
                st.warning(f"No question mapping available for feature: {feature}")
                # Default to medium level if we must proceed
                user_input[feature] = 1
    
    return user_input

# -----------------------------
# Welcome Screen
# -----------------------------
import base64
import streamlit as st
from pathlib import Path

def show_welcome_screen():
    # Convert local image to base64
    def get_image_base64(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except FileNotFoundError:
            st.error(f"Image file not found at: {image_path}")
            return None
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None
    
    # Try multiple possible image locations
    image_paths = [
       "Image.jpeg",  # Current directory
        r"C:\Users\PMYLS\Documents\Downloads\Image.jpeg",  # Original path
        "assets/Image.jpeg",  # Common assets folder
        "resources/Image.jpeg"  # Another common location
    ]
    
    image_base64 = None
    for path in image_paths:
        image_base64 = get_image_base64(path)
        if image_base64 is not None:
            break
    
    # Main container for layout
    main_container = st.container()
    
    if image_base64 is None:
        st.error("Could not load background image. Using plain background.")
        st.markdown(
            """
            <style>
                .stApp {
                    background-color: #f0f2f6;
                    height: 100vh;
                    overflow: hidden;
                }
                .main .block-container {
                    padding: 0;
                    height: 100vh;
                    position: relative;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        # Set background image and disable scrolling
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-image: url("data:image/jpeg;base64,{image_base64}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                    height: 100vh;
                    overflow: hidden;
                    margin: 0;
                    padding: 0;
                }}
                html, body {{
                    height: 100%;
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                }}
                .main .block-container {{
                    padding: 0;
                    height: 100vh;
                    position: relative;
                    max-width: 100%;
                }}
                .button-container {{
                    position: fixed;
                    bottom: 20px;
                    left: 0;
                    right: 0;
                    display: flex;
                    justify-content: center;
                    z-index: 999;
                    width: 100%;
                }}
                .stButton {{
                    display: flex;
                    justify-content: center;
                    width: 100%;
                }}
                .stButton>button {{
                    padding: 15px 30px;
                    font-size: 20px;
                    border-radius: 30px;
                    background-color: white;
                    color: #800080;
                    border: 2px solid #800080;
                    cursor: pointer;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    transition: all 0.3s;
                    width: 250px;
                    font-weight: bold;
                }}
                .stButton>button:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
                    background-color: #f0e6ff;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
    # Add content to main container
    with main_container:
        st.markdown(
            """
            <div style='text-align: center; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
                <h1>   </h1>
                <h2>    </h2>
                <h2>    </h2>
                <h2>   </h2>
                <h3>     </h3>
                <p>       </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Create the button (it will be centered due to our CSS)
    if st.button("Start Assessment", key="start_button"):
        st.session_state.show_assessment = True
        st.rerun()


    # Hide the Streamlit footer and header
    hide_streamlit_style = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -----------------------------
# Assessment Screen
# -----------------------------
def show_assessment_screen():
    # Set background image for assessment screen
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #564172;

        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Header with logo
    with st.container():
        st.markdown("""
        <div class="main-container">
            <div class="header">
                <div class="title">🧭 Career Path Predictor</div>
                <div class="subtitle">
                    Discover your ideal career based on your skills, preferences, and personality
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()

    # Show raw data sample
    if st.checkbox("Show raw data sample", key="show_data"):
        with st.container():
            st.markdown("""
            <div class="main-container">
                <h2 class="section-header">Data Overview</h2>
            """, unsafe_allow_html=True)
            st.dataframe(df.head().style.set_properties(**{
                'background-color': '#f8f9fa',
                'color': '#212529',
                'border': '1px solid #dee2e6'
            }))
            st.markdown("</div>", unsafe_allow_html=True)

    # Enhanced preprocessing
    df_processed, le_dict, target_le, category_mapping = preprocess_data(df)

    # Prepare features and target
    X = df_processed.drop("Predicted_Career_Field", axis=1)
    y = df_processed["Predicted_Career_Field"]

    # Train model with feature selection (keeping top 30 features)
    model, selected_features = train_model(X, y, n_features=30)
    
    with st.container():
        st.markdown("""
        <div class="main-container">
            <h2 class="section-header">Career Assessment Questionnaire</h2>
            <div style="color: #7f8c8d; margin-bottom: 2rem;">
                Please answer the following questions honestly. Your responses will help us determine 
                the career path that best matches your skills and preferences.
            </div>
        """, unsafe_allow_html=True)

        # Get user input - only for selected features
        with st.form("career_form"):
            user_input = ask_questions(selected_features)
            
            # Form submit button only (reset button removed)
            submit_button = st.form_submit_button("🔮 Predict My Career", type="primary")
            
        st.markdown("</div>", unsafe_allow_html=True)

    # Handle form submission
    if submit_button:
        if len(user_input) == len(selected_features):
            # Convert user input to DataFrame with selected features
            input_df = pd.DataFrame([user_input], columns=selected_features)
            
            # Enhanced encoding handling
            for col in input_df.columns:
                if col in le_dict:
                    # Handle categorical features that need encoding
                    if isinstance(user_input[col], str):
                        try:
                            input_df[col] = le_dict[col].transform([user_input[col]])[0]
                        except ValueError:
                            # If value wasn't in training data, use most common category
                            st.warning(f"Note: Unseen value '{user_input[col]}' for {col} was mapped to default")
                            input_df[col] = le_dict[col].transform([le_dict[col].classes_[0]])[0]
                elif isinstance(user_input[col], str) and col not in ['Field_of_Study', 'Highest_Degree', 
                                                                     'Work_Hour_Flexibility', 'Location_Preference',
                                                                     'Industry_of_Experience']:
                    # Default for other string features not in le_dict
                    input_df[col] = 0
            
            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                predicted_career = target_le.inverse_transform([prediction])[0]
                
                with st.container():
                    st.markdown(f"""
                    <div class="main-container">
                        <div class="prediction-card">
                            <h2 style="color: white; margin-bottom: 1rem; font-size: 1.8rem;">Your Career Prediction</h2>
                            <p style="font-size: 2rem; font-weight: bold; margin-bottom: 0;">{predicted_career}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show additional insights
                    with st.expander("📊 Show prediction details", expanded=False):
                        st.markdown("""
                        <div style="margin-top: 1.5rem;">
                            <h3 class="section-header">Your Input Summary</h3>
                        """, unsafe_allow_html=True)
                        
                        # Create a readable version of user input for display
                        display_input = pd.DataFrame.from_dict(user_input, orient='index', columns=['Value'])
                        st.dataframe(display_input.style.set_properties(**{
                            'background-color': '#f8f9fa',
                            'color': '#212529',
                            'border': '1px solid #dee2e6'
                        }))
                        
                        st.markdown("""
                        <h3 class="section-header">Top Features Influencing Your Prediction</h3>
                        """, unsafe_allow_html=True)
                        feature_importances = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        # Only show features with importance > 0
                        feature_importances = feature_importances[feature_importances['Importance'] > 0]
                        
                        st.dataframe(feature_importances.style.background_gradient(
                            cmap='Blues', subset=['Importance']
                        ).set_properties(**{
                            'background-color': '#f8f9fa',
                            'color': '#212529',
                            'border': '1px solid #dee2e6'
                        }))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                with st.container():
                    st.markdown("""
                    <div class="main-container">
                    """, unsafe_allow_html=True)
                    st.error(f"An error occurred during prediction: {str(e)}")
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown("""
                <div class="main-container">
                """, unsafe_allow_html=True)
                st.error("Please answer all questions before predicting.")
                st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Main App
# -----------------------------
def main():
    # Initialize session state for page navigation
    if 'show_assessment' not in st.session_state:
        st.session_state.show_assessment = False
    
    if not st.session_state.show_assessment:
        show_welcome_screen()
    else:
        show_assessment_screen()

if __name__ == "__main__":
    main()# ====================== IMPORTS ======================
import matplotlib.pyplot as plt  # For data visualization
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import streamlit as st  # For building the web app interface
from sklearn.tree import DecisionTreeClassifier, export_text  # Machine learning model
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables
from sklearn.metrics import accuracy_score  # For evaluating model performance
import random  # For randomizing questions

# ====================== STYLING & SETUP ======================
# Configure the Streamlit page settings
st.set_page_config(
    page_title="AI Powered Career Prediction Based on Personality Traits",  # Browser tab title
    page_icon="🧭",  # Browser tab icon
    layout="wide",  # Use wider page layout
    initial_sidebar_state="expanded"  # Start with sidebar expanded
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
@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_data():
    """Loads and preprocesses the career prediction dataset"""
    career_options = [
        # Comprehensive list of potential career options
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
    """Preprocesses the data by encoding categorical variables"""
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
    return data, le  # Return processed data and the target encoder

# Process the data
processed_data, target_le = preprocess_data(data.copy())

def train_model(data):
    """Trains a decision tree classifier on the processed data"""
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
    """Selects 10 random questions from the pool of 30, ensuring diversity"""
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

def generate_career_insights(predicted_career, user_responses, top_features):
    """Generates detailed insights about the predicted career match"""
    # Career descriptions
    career_descriptions = {
        "Software Developer": "focused on creating and maintaining software applications",
        "Data Scientist": "working with data to extract insights and build predictive models",
        "AI Engineer": "developing artificial intelligence systems and machine learning models",
        # Add descriptions for all other careers
        "default": "that aligns well with your skills and personality"
    }
    
    # Generate the prediction paragraph
    paragraph = f"""
    Based on your assessment, you would excel as a **{predicted_career}**, \
    {career_descriptions.get(predicted_career, career_descriptions['default'])}. \
    Your unique combination of skills and preferences makes this an excellent match.
    """
    
    # Generate summary points
    summary = []
    for feat in top_features.index:
        if feat == "Interest":
            interest = user_responses.get("Interest", "diverse")
            summary.append(f"Your interest in {interest} fields matches this career path")
        elif feat == "Work_Style":
            style = user_responses.get("Work_Style", "working style")
            summary.append(f"Your preference for {style.lower()} work environments fits well")
        elif feat == "Strengths":
            strength = user_responses.get("Strengths", "strengths")
            summary.append(f"Your {strength.lower()} abilities are valuable in this field")
        elif feat == "GPA":
            gpa = user_responses.get("GPA", 3.0)
            summary.append(f"Your academic performance (GPA: {gpa}) meets typical requirements")
        elif feat == "Years_of_Experience":
            exp = user_responses.get("Years_of_Experience", 0)
            summary.append(f"Your {exp} years of experience provide a solid foundation")
        else:
            summary.append(f"Your {feat.replace('_', ' ').lower()} aligns with this career")
    
    # Identify key traits
    traits = []
    if user_responses.get("Communication_Skills") == "High":
        traits.append("Strong communicator")
    if user_responses.get("Leadership_Skills") == "High":
        traits.append("Leadership potential")
    if user_responses.get("Teamwork_Skills") == "High":
        traits.append("Team player")
    if user_responses.get("GPA", 0) > 3.5:
        traits.append("Academic achiever")
    if user_responses.get("Years_of_Experience", 0) > 5:
        traits.append("Experienced professional")
    
    # Generate suggestions
    suggestions = [
        f"Research educational requirements for {predicted_career} positions",
        "Identify key skills to develop for this career path",
        "Connect with professionals currently working in this field",
        "Look for internships or entry-level positions to gain experience",
        "Consider relevant certifications or additional training"
    ]
    
    return {
        "paragraph": paragraph,
        "summary": summary,
        "traits": traits,
        "suggestions": suggestions
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

        # Prediction and results section
        if st.button("🔮 Find My Career Match"):
            try:
                # Prepare input data for prediction
                input_data = pd.DataFrame([st.session_state.user_responses])
                
                # Convert categorical features using LabelEncoder
                processed_input = input_data.copy()
                for col in processed_data.columns:
                    if col in input_data.columns and col != 'Predicted_Career_Field':
                        le = LabelEncoder()
                        le.fit(processed_data[col])
                        processed_input[col] = le.transform(input_data[col])

                # Make prediction
                prediction = model.predict(processed_input)
                predicted_career = target_le.inverse_transform(prediction)[0]
                
                # Get feature importances
                feat_importances = pd.Series(model.feature_importances_, 
                                           index=processed_input.columns)
                top_features = feat_importances.sort_values(ascending=False).head(3)

                # Generate insights
                insights = generate_career_insights(predicted_career, 
                                                  st.session_state.user_responses, 
                                                  top_features)

                # Display results in a structured format
                with st.container():
                    st.markdown(f"""
                    <div style="background: #f8fafc; padding: 2rem; border-radius: 12px; 
                                border-left: 5px solid #4a90e2; margin-bottom: 2rem;">
                        <h2 style="color: #2d3748; margin-top: 0;">Your Career Match: {predicted_career}</h2>
                        <p style="font-size: 1.1rem;">{insights['paragraph']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Why this career fits you
                    with st.expander("🔍 Why this career matches your profile", expanded=True):
                        st.markdown("""
                        <div style="background: white; padding: 1.5rem; border-radius: 8px;">
                            <ul style="margin-top: 0;">
                        """ + "\n".join([f"<li>{point}</li>" for point in insights['summary']]) + """
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Feature importance visualization
                        fig, ax = plt.subplots(figsize=(8, 4))
                        top_features.sort_values().plot(kind='barh', color='#4a90e2', ax=ax)
                        ax.set_title('Key Factors in Your Career Match')
                        ax.set_xlabel('Importance Score')
                        st.pyplot(fig)
                    
                    # Your key traits
                    if insights['traits']:
                        st.subheader("🌟 Your Key Traits")
                        cols = st.columns(4)
                        for i, trait in enumerate(insights['traits']):
                            cols[i % 4].markdown(f"""
                            <div style="background: #ebf8ff; color: #2b6cb0; 
                                        padding: 0.5rem 1rem; border-radius: 20px; 
                                        text-align: center; margin-bottom: 0.5rem;">
                                {trait}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Next steps
                    st.subheader("🚀 Next Steps to Pursue This Career")
                    for i, suggestion in enumerate(insights['suggestions'], 1):
                        st.markdown(f"""
                        <div style="display: flex; align-items: flex-start; margin-bottom: 0.5rem;">
                            <div style="background: #2b6cb0; color: white; width: 24px; height: 24px; 
                                        border-radius: 50%; display: flex; align-items: center; 
                                        justify-content: center; margin-right: 0.5rem; flex-shrink: 0;">
                                {i}
                            </div>
                            <div style="flex-grow: 1;">
                                {suggestion}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                # Career details expander
                with st.expander("📚 Learn more about this career"):
                    st.write(f"## {predicted_career} Career Overview")
                    st.write("**Typical Responsibilities:**")
                    st.write("- Develop and maintain software systems")
                    st.write("- Collaborate with cross-functional teams")
                    st.write("- Implement new features and improvements")
                    st.write("\n**Average Salary Range:** $80,000 - $120,000")
                    st.write("\n**Required Education:** Bachelor's degree in Computer Science or related field")

            except Exception as e:
                st.error(f"We encountered an issue analyzing your profile. Please try again.")
                st.error(str(e))

if __name__ == "__main__":
    main()
