import streamlit as st
import requests
import json
import sqlite3
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken


st.set_page_config(page_title="LLM Benchmarking for Accounting", page_icon="💼", layout="wide")

# Function to analyze model performance using ML techniques
def analyze_model_performance(conn):
    """
    Analyze model performance using machine learning techniques
    """
    # Get comprehensive quiz data
    query = """
    SELECT 
        r.model_name,
        r.accuracy, 
        r.completeness, 
        r.clarity, 
        r.total_score,
        ques.difficulty
    FROM responses r
    JOIN questions ques ON r.question_id = ques.question_id
    WHERE r.accuracy IS NOT NULL
    AND NOT r.response_text LIKE 'Error:%'
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        
        if len(df) < 10:
            return None, "Not enough data for meaningful analysis. Need at least 10 responses."
            
        # Prepare the data
        # Encode difficulty levels
        df['difficulty_encoded'] = df['difficulty'].map({'Easy': 1, 'Medium': 2, 'Hard': 3})
        
        # Feature set for analysis
        features = ['accuracy', 'completeness', 'clarity', 'difficulty_encoded']
        
        # Get unique models
        models = df['model_name'].unique()
        
        results = {
            'overall_best': None,
            'difficulty_analysis': {},
            'feature_importance': {},
            'clustering': None,
            'pca_data': None,
            'pca_models': None
        }
        
        # 1. Overall best model by average score
        avg_scores = df.groupby('model_name')['total_score'].mean().sort_values(ascending=False)
        results['overall_best'] = avg_scores.index[0]
        results['avg_scores'] = avg_scores
        
        # 2. Best model by difficulty
        for difficulty in ['Easy', 'Medium', 'Hard']:
            diff_df = df[df['difficulty'] == difficulty]
            if len(diff_df) > 0:
                diff_scores = diff_df.groupby('model_name')['total_score'].mean().sort_values(ascending=False)
                if len(diff_scores) > 0:
                    results['difficulty_analysis'][difficulty] = diff_scores
        
        # 3. Feature importance by model
        for feature in ['accuracy', 'completeness', 'clarity']:
            feature_scores = df.groupby('model_name')[feature].mean().sort_values(ascending=False)
            results['feature_importance'][feature] = feature_scores
            
        # 4. Clustering models by performance
        if len(df) >= 10:
            # Prepare data for clustering
            model_features = df.groupby('model_name')[features].mean().reset_index()
            if len(model_features) > 1:  # Need at least 2 models for meaningful clustering
                X = model_features[features].values
                
                # Standardize the features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Determine optimal number of clusters (max 3 or num_models)
                max_clusters = min(3, len(models))
                
                if max_clusters > 1:  # Need at least 2 clusters
                    # Apply KMeans
                    kmeans = KMeans(n_clusters=max_clusters, random_state=42)
                    model_features['cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # Store clustering results
                    results['clustering'] = model_features
                    
                    # 5. Apply PCA for visualization
                    pca = PCA(n_components=2)
                    principal_components = pca.fit_transform(X_scaled)
                    
                    # Store PCA results
                    results['pca_data'] = principal_components
                    results['pca_models'] = model_features['model_name'].values
                    results['pca_variance'] = pca.explained_variance_ratio_
                    
        return results, None
    except Exception as e:
        return None, f"Error in ML analysis: {str(e)}"

# Function to generate accounting questions using LLM
def generate_accounting_questions(evaluator_model):
    prompt = """
You are a expert accounting educator and professional assessment designer. Your task is to generate exactly three unique accounting questions—one at each difficulty level (Easy, Medium, Hard)—with fully developed answers. Follow these rules to maximize quality:

1. **Distinct Domains**  
   - Easy: fundamental concept or definition.  
   - Medium: application or analysis (e.g., journal entries, adjusting entries, financial statement preparation, ratio analysis).  
   - Hard: evaluation or synthesis on advanced topics (e.g., complex revenue recognition under ASC 606/IFRS 15, lease accounting, business combinations, tax implications).

2. **Question Style**  
   - Easy: straightforward "what is" or "identify" question.  
   - Medium & Hard: scenario-based context that requires critical thinking and multi-step reasoning.

3. **Answer Depth**  
   - Provide at least three sentences.  
   - Explain the rationale, reference key standards or principles, and highlight common pitfalls.

4. **Output Format**  
   - A single, valid JSON array of three objects, ordered from Easy to Hard.  
   - No extra text, markdown, code fences, or comments—only the raw JSON array.

5. **Schema**  
   Each object must have exactly these keys (no extras):
   {
     "difficulty": "Easy" | "Medium" | "Hard",
     "question": <string>,
     "answer": <string>
   }

Ensure the JSON is syntactically correct and complete.
"""
    
    system_prompt = "You are a JSON generator that only outputs valid, well-formatted JSON. Your responses will be directly parsed as JSON objects with no additional processing. Never include explanatory text or anything outside the requested JSON structure."
    
    try:
        response_data = query_ollama(evaluator_model, prompt, system_prompt)
        response = response_data["response"]
        
        # Check if response is an error message
        if response.startswith("Error:"):
            st.error(f"LLM error: {response}")
            return fallback_questions()
        
        # Try different JSON parsing approaches
        try:
            # First try: direct JSON parsing of the whole response
            questions = json.loads(response)
            if isinstance(questions, list) and len(questions) > 0:
                # Verify we have 3 questions with proper fields
                if len(questions) == 3 and all("difficulty" in q and "question" in q and "answer" in q for q in questions):
                    return questions
                else:
                    st.warning(f"LLM returned {len(questions)} questions instead of 3 or missing required fields")
        except json.JSONDecodeError:
            # Second try: extract JSON using bracket finding
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                try:
                    json_str = response[json_start:json_end]
                    questions = json.loads(json_str)
                    if isinstance(questions, list) and len(questions) > 0:
                        # Verify we have 3 questions with proper fields
                        if len(questions) == 3 and all("difficulty" in q and "question" in q and "answer" in q for q in questions):
                            return questions
                        else:
                            st.warning(f"LLM returned {len(questions)} questions instead of 3 or missing required fields")
                except json.JSONDecodeError:
                    pass  # Continue to fallback
        
        # If we got here, JSON parsing failed - log the response for debugging
        st.warning("LLM did not generate properly formatted questions. Using fallback questions.")
        st.error(f"Invalid JSON response: {response[:100]}...")  # Show first 100 chars for debugging
        return fallback_questions()
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return fallback_questions()

# Fallback questions if LLM generation fails
def fallback_questions():
    return [
        {
            "difficulty": "Easy",
            "question": "What is the accounting equation and explain its components?",
            "answer": "The accounting equation is Assets = Liabilities + Equity. Assets are resources owned by a business. Liabilities are obligations or debts owed to others. Equity represents the owner's interest in the business."
        },
        {
            "difficulty": "Medium",
            "question": "Explain the concept of depreciation and the different methods of calculating it.",
            "answer": "Depreciation is the systematic allocation of an asset's cost over its useful life. Methods include: Straight-line (equal amounts each period), Declining balance (accelerated depreciation with higher amounts in earlier years), Units of production (based on actual usage), and Sum-of-the-years'-digits (accelerated method based on remaining useful life)."
        },
        {
            "difficulty": "Hard",
            "question": "Explain the concept of deferred tax assets and liabilities in accordance with accounting standards.",
            "answer": "Deferred tax assets and liabilities arise from temporary differences between accounting income and taxable income. Deferred tax assets represent future tax benefits (from deductible temporary differences) while deferred tax liabilities represent future tax obligations (from taxable temporary differences). They're recognized when there's a difference between the tax basis of assets/liabilities and their carrying amounts in financial statements."
        }
    ]

# Database functions
def init_db():
    if not os.path.exists('db'):
        os.makedirs('db')
    
    conn = sqlite3.connect('db/quiz_results.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''
    CREATE TABLE IF NOT EXISTS quizzes (
        quiz_id INTEGER PRIMARY KEY,
        timestamp TEXT,
        evaluator_model TEXT
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS questions (
        question_id INTEGER PRIMARY KEY,
        difficulty TEXT,
        question_text TEXT,
        correct_answer TEXT
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS responses (
        response_id INTEGER PRIMARY KEY,
        quiz_id INTEGER,
        question_id INTEGER,
        model_name TEXT,
        response_text TEXT,
        accuracy INTEGER,
        completeness INTEGER,
        clarity INTEGER,
        total_score INTEGER,
        feedback TEXT,
        evaluator_model TEXT,
        response_time REAL,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        final_score REAL,
        FOREIGN KEY (quiz_id) REFERENCES quizzes (quiz_id),
        FOREIGN KEY (question_id) REFERENCES questions (question_id)
    )
    ''')
    
    # Check if we need to add the new columns to an existing table
    c.execute("PRAGMA table_info(responses)")
    columns = c.fetchall()
    column_names = [column[1] for column in columns]
    
    # Add response_time column if it doesn't exist
    if 'response_time' not in column_names:
        c.execute("ALTER TABLE responses ADD COLUMN response_time REAL")
    
    # Add token columns if they don't exist
    if 'prompt_tokens' not in column_names:
        c.execute("ALTER TABLE responses ADD COLUMN prompt_tokens INTEGER")
    
    if 'completion_tokens' not in column_names:
        c.execute("ALTER TABLE responses ADD COLUMN completion_tokens INTEGER")
    
    if 'total_tokens' not in column_names:
        c.execute("ALTER TABLE responses ADD COLUMN total_tokens INTEGER")
    
    # Add final_score column if it doesn't exist
    if 'final_score' not in column_names:
        c.execute("ALTER TABLE responses ADD COLUMN final_score REAL")
    
    conn.commit()
    conn.close()

def verify_admin(username, password):
    # Simple hardcoded admin credentials instead of database lookup
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    return username == admin_username and password == admin_password

def create_new_quiz(evaluator_model):
    conn = sqlite3.connect('db/quiz_results.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO quizzes (timestamp, evaluator_model) VALUES (?, ?)", (timestamp, evaluator_model))
    quiz_id = c.lastrowid
    conn.commit()
    conn.close()
    return quiz_id

def save_response(quiz_id, question_text, model_name, response_text, evaluation, evaluator_model):
    try:
        conn = sqlite3.connect('db/quiz_results.db')
        c = conn.cursor()
        
        # Get question ID or insert if not exists
        c.execute("SELECT question_id FROM questions WHERE question_text = ?", (question_text,))
        result = c.fetchone()
        
        if result:
            question_id = result[0]
        else:
            # This is a new question generated by LLM, insert it
            for question in st.session_state.accounting_questions:
                if question["question"] == question_text:
                    c.execute("INSERT INTO questions (difficulty, question_text, correct_answer) VALUES (?, ?, ?)",
                            (question["difficulty"], question_text, question["answer"]))
                    question_id = c.lastrowid
                    break
            else:
                # Fallback if not found
                c.execute("INSERT INTO questions (difficulty, question_text, correct_answer) VALUES (?, ?, ?)",
                        ("Unknown", question_text, ""))
                question_id = c.lastrowid
        
        # Calculate final_score if not already present in evaluation
        final_score = evaluation.get("final_score", 0.0)
        
        # Save response
        c.execute('''
        INSERT INTO responses (
            quiz_id, question_id, model_name, response_text, 
            accuracy, completeness, clarity, total_score, feedback, evaluator_model, 
            response_time, prompt_tokens, completion_tokens, total_tokens, final_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            quiz_id, 
            question_id, 
            model_name, 
            response_text, 
            evaluation.get("accuracy", 0), 
            evaluation.get("completeness", 0), 
            evaluation.get("clarity", 0), 
            evaluation.get("total_score", 0), 
            evaluation.get("feedback", ""),
            evaluator_model,
            evaluation.get("response_time", 0),
            evaluation.get("prompt_tokens", 0),
            evaluation.get("completion_tokens", 0),
            evaluation.get("total_tokens", 0),
            final_score
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error saving to database: {e}")

def get_tiktoken_encoding(model_name):
    """
    Get the tiktoken encoding for token counting.
    Using cl100k_base for all models as a standard tokenizer.
    """
    # Use cl100k_base for all models
    return tiktoken.get_encoding("cl100k_base")

# Helper function to get question_id from question_text
def question_id_for_text(conn, question_text):
    c = conn.cursor()
    c.execute("SELECT question_id FROM questions WHERE question_text = ?", (question_text,))
    result = c.fetchone()
    if result:
        return result[0]
    return None

# API function
def query_ollama(model, prompt, system=""):
    try:
        # Set longer timeout for larger models
        timeout = 2000 if "gemma3:12b" in model.lower() else 120
        
        # Record start time
        start_time = datetime.now()
        
        # Calculate prompt token count using tiktoken
        encoding = get_tiktoken_encoding(model)
        prompt_tokens = len(encoding.encode(prompt))
        system_tokens = len(encoding.encode(system)) if system else 0
        total_prompt_tokens = prompt_tokens + system_tokens
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": False
            },
            timeout=timeout  # 5 minutes for gemma3:12b, 2 minutes for others
        )
        response.raise_for_status()
        
        # Get response text
        response_text = response.json()["response"]
        
        # Calculate response time in seconds
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Calculate response token count
        completion_tokens = len(encoding.encode(response_text))
        
        # Return response text and metadata
        return {
            "response": response_text,
            "response_time": response_time,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_prompt_tokens + completion_tokens
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "response_time": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

def evaluate_response(question, correct_answer, model_response, evaluator_model):
    # Skip evaluation if the response is an error message
    if model_response["response"].startswith("Error:"):
        return {
            "accuracy": 0,
            "completeness": 0,
            "clarity": 0,
            "total_score": 0,
            "feedback": "Response contained an error and could not be evaluated.",
            "response_time": model_response.get("response_time", 0),
            "prompt_tokens": model_response.get("prompt_tokens", 0),
            "completion_tokens": model_response.get("completion_tokens", 0),
            "total_tokens": model_response.get("total_tokens", 0)
        }
        
    evaluation_prompt = f"""
    You are an expert accounting professor evaluating a response to an accounting question.
    
    Question: {question}
    
    Correct answer concepts: {correct_answer}
    
    Model's response: {model_response["response"]}
    
    Please evaluate the response on a scale of 0-10 based on:
    1. Accuracy (0-10): How factually correct is the response?
    2. Completeness (0-10): How thoroughly does it address all aspects of the question?
    3. Clarity (0-10): How clear and well-structured is the explanation?
    
    Provide your evaluation in JSON format with the following structure:
    {{
        "accuracy": score,
        "completeness": score,
        "clarity": score,
        "total_score": sum_of_scores,
        "feedback": "brief explanation of the evaluation"
    }}
    
    Return ONLY the JSON object, nothing else.
    """
    
    try:
        evaluation_result = query_ollama(evaluator_model, evaluation_prompt)
        # Extract JSON from the response
        json_start = evaluation_result["response"].find('{')
        json_end = evaluation_result["response"].rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = evaluation_result["response"][json_start:json_end]
            evaluation = json.loads(json_str)
            # Add response time and token count from the original response
            evaluation["response_time"] = model_response.get("response_time", 0)
            evaluation["prompt_tokens"] = model_response.get("prompt_tokens", 0)
            evaluation["completion_tokens"] = model_response.get("completion_tokens", 0)
            evaluation["total_tokens"] = model_response.get("total_tokens", 0)
            return evaluation
        else:
            return {
                "accuracy": 5,
                "completeness": 5,
                "clarity": 5,
                "total_score": 15,
                "feedback": "Error parsing evaluation",
                "response_time": model_response.get("response_time", 0),
                "prompt_tokens": model_response.get("prompt_tokens", 0),
                "completion_tokens": model_response.get("completion_tokens", 0),
                "total_tokens": model_response.get("total_tokens", 0)
            }
    except Exception:
        return {
            "accuracy": 5,
            "completeness": 5,
            "clarity": 5,
            "total_score": 15,
            "feedback": "Error during evaluation",
            "response_time": model_response.get("response_time", 0),
            "prompt_tokens": model_response.get("prompt_tokens", 0),
            "completion_tokens": model_response.get("completion_tokens", 0),
            "total_tokens": model_response.get("total_tokens", 0)
        }

# Initialize session state
if 'scores' not in st.session_state:
    st.session_state.scores = {}
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'quiz_complete' not in st.session_state:
    st.session_state.quiz_complete = False
if 'models_to_test' not in st.session_state:
    st.session_state.models_to_test = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'quiz_id' not in st.session_state:
    st.session_state.quiz_id = None
if 'accounting_questions' not in st.session_state:
    st.session_state.accounting_questions = []
if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'current_model_index' not in st.session_state:
    st.session_state.current_model_index = 0
if 'questions_generated' not in st.session_state:
    st.session_state.questions_generated = False

# Initialize database
init_db()

# Main app
st.title("🧮 LLM Benchmarking for Accounting")
st.markdown("This tool benchmarks large language models (LLMs) on their accounting knowledge, evaluating responses for accuracy, completeness, and clarity.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Get available models
    try:
        models_response = requests.get("http://localhost:11434/api/tags")
        available_models = [model["name"] for model in models_response.json()["models"]]
    except:
        available_models = ["llama3", "mistral", "gemma"]
        st.warning("Could not fetch models from Ollama. Using default list.")
    
    # Model selection
    st.subheader("Select Models to Benchmark")
    model_selections = {}
    for model in available_models:
        model_selections[model] = st.checkbox(model, value=model in ["llama3", "mistral"] if len(st.session_state.models_to_test) == 0 else model in st.session_state.models_to_test)
    
    # Evaluator model
    evaluator_model = st.selectbox(
        "Select Evaluator Model",
        available_models,
        index=available_models.index("llama3") if "llama3" in available_models else 0
    )
    
    # Start button
    if st.button("Start Benchmark"):
        st.session_state.models_to_test = [model for model, selected in model_selections.items() if selected]
        st.session_state.evaluator_model = evaluator_model
        st.session_state.scores = {model: 0 for model in st.session_state.models_to_test}
        st.session_state.responses = {}
        st.session_state.quiz_complete = False
        st.session_state.processing = True
        st.session_state.quiz_id = create_new_quiz(evaluator_model)
        st.session_state.questions_generated = False
        st.session_state.current_question_index = 0
        st.session_state.current_model_index = 0
        
        # Generate questions with the evaluator model
        with st.spinner("Generating accounting questions..."):
            st.session_state.accounting_questions = generate_accounting_questions(evaluator_model)
            st.session_state.questions_generated = True
        
        st.rerun()
    
    # Database stats
    st.subheader("Database Statistics")
    try:
        conn = sqlite3.connect('db/quiz_results.db')
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM quizzes")
        num_quizzes = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM responses")
        num_responses = c.fetchone()[0]
        conn.close()
        
        st.markdown(f"**Total benchmarks run:** {num_quizzes}")
        st.markdown(f"**Total model responses:** {num_responses}")
    except Exception:
        st.warning("Database statistics not available.")
    
    # Admin login section - moved to bottom of sidebar
    st.markdown("---")
    st.subheader("Admin Login")
    
    if not st.session_state.admin_logged_in:
        with st.form("admin_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if verify_admin(username, password):
                    st.session_state.admin_logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    else:
        st.success("Admin logged in")
        
        # Add debug option for question generation
        with st.expander("🛠️ Admin Debug Tools"):
            test_model = st.selectbox(
                "Select model for testing",
                available_models,
                index=available_models.index("llama3") if "llama3" in available_models else 0
            )
            
            if st.button("Test Question Generation"):
                with st.spinner("Testing question generation..."):
                    test_questions = generate_accounting_questions(test_model)
                    
                    st.subheader("Generated Questions")
                    if test_questions and len(test_questions) > 0:
                        st.success(f"Successfully generated {len(test_questions)} questions!")
                        st.json(test_questions)
                    else:
                        st.error("Failed to generate questions, fallback used")
                        st.json(test_questions)
        
        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

# Main content
if st.session_state.admin_logged_in and len(st.session_state.models_to_test) == 0:
    # Show Database Review Interface in main content when admin is logged in and no quiz is running
    tabs = st.tabs(["SQL Database Review", "ML Model Analysis"])
    
    # SQL Database Review Tab
    with tabs[0]:
        st.header("SQL Database Review")
        st.markdown("**Run SQL queries directly on the database**")
        
        # Sample queries dropdown
        sample_queries = {
            "Select a sample query": "",
            "All tables in database": "SELECT name FROM sqlite_master WHERE type='table';",
            "All quiz results": """
                    SELECT 
                        q.quiz_id, q.timestamp, q.evaluator_model,
                        ques.question_id, ques.difficulty, ques.question_text,
                        r.response_id, r.model_name, r.response_text, 
                        r.accuracy, r.completeness, r.clarity, r.total_score, r.feedback,
                        r.response_time, r.prompt_tokens, r.completion_tokens, r.total_tokens, r.final_score
                    FROM quizzes q
                    JOIN responses r ON q.quiz_id = r.quiz_id
                    JOIN questions ques ON r.question_id = ques.question_id
            ORDER BY q.timestamp DESC LIMIT 100;
            """,
            "Quiz performance by model": """
            SELECT 
                r.model_name, 
                AVG(r.total_score) as avg_score,
                COUNT(*) as total_responses
            FROM responses r
            GROUP BY r.model_name
            ORDER BY avg_score DESC;
            """,
            "Questions by difficulty": """
            SELECT difficulty, COUNT(*) as count
            FROM questions
            GROUP BY difficulty
            ORDER BY 
                CASE difficulty 
                    WHEN 'Easy' THEN 1 
                    WHEN 'Medium' THEN 2 
                    WHEN 'Hard' THEN 3 
                    ELSE 4 
                END;
            """
        }
        
        selected_sample = st.selectbox("Sample queries", options=list(sample_queries.keys()))
        
        # SQL query input
        sql_query = st.text_area(
            "Enter SQL query", 
            value=sample_queries[selected_sample],
            height=150,
            help="Enter your SQL query here. SELECT queries only for safety."
        )
        
        # Safety check for non-SELECT queries
        is_select = sql_query.strip().upper().startswith("SELECT")
        is_pragma = sql_query.strip().upper().startswith("PRAGMA")
        
        if st.button("Run Query"):
            if not sql_query:
                st.warning("Please enter a SQL query")
            elif not (is_select or is_pragma):
                st.error("Only SELECT and PRAGMA queries are allowed for safety reasons")
            else:
                try:
                    conn = sqlite3.connect('db/quiz_results.db')
                    
                    # Execute query and convert to DataFrame
                    results_df = pd.read_sql_query(sql_query, conn)
                    conn.close()
                    
                    # Display results
                    st.subheader("Query Results")
                    st.dataframe(
                        results_df, 
                        use_container_width=True,
                        hide_index=False
                    )
                    
                    # Show row count
                    st.info(f"Retrieved {len(results_df)} records")
                    
                    # Export option
                    if not results_df.empty:
                        # Create CSV data for download
                        csv_data = results_df.to_csv(index=False)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"query_results_{timestamp}.csv"
                        
                        # Use Streamlit's download button instead of regular button
                        st.download_button(
                            label="Export Results to CSV",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error executing query: {e}")
    
    # ML Model Analysis Tab
    with tabs[1]:
        st.header("ML Model Analysis")
        st.markdown("**Machine Learning Analysis to Determine the Best Model for Accounting**")
        
        if st.button("Run Analysis"):
            with st.spinner("Running ML analysis on benchmark results..."):
                try:
                    conn = sqlite3.connect('db/quiz_results.db')
                    
                    # First check if we have enough data
                    c = conn.cursor()
                    c.execute("SELECT COUNT(*) FROM responses WHERE accuracy IS NOT NULL")
                    count = c.fetchone()[0]
                    
                    if count < 10:
                        st.warning("Not enough data for analysis. Need at least 10 quiz responses.")
                    else:
                        # Run the ML analysis
                        results, error = analyze_model_performance(conn)
                        
                        if error:
                            st.error(error)
                        else:
                            # Display overall best model
                            st.subheader("Overall Best Model")
                            best_model = results['overall_best']
                            best_score = results['avg_scores'][best_model]
                            
                            # Create a fancy metric display for the best model
                            st.markdown(f"""
                            <div style='background-color: #f0f7ff; padding: 20px; border-radius: 10px; border: 1px solid #d0e3ff;'>
                                <h2 style='text-align: center; margin-bottom: 10px;'>{best_model}</h2>
                                <h3 style='text-align: center; color: #1f77b4;'>Score: {best_score:.2f}/30</h3>
                                <p style='text-align: center;'>🏆 Best Overall Model for Accounting</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display average scores for all models
                            st.subheader("Average Performance by Model")
                            
                            # Convert scores to dataframe for charting
                            avg_scores_df = results['avg_scores'].reset_index()
                            avg_scores_df.columns = ['Model', 'Average Score']
                            
                            # Display as bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x='Model', y='Average Score', data=avg_scores_df, ax=ax, palette='viridis')
                            ax.set_title('Average Model Performance')
                            ax.set_ylim(0, 30)  # Max score is 30
                            st.pyplot(fig)
                            
                            # Display performance by difficulty
                            st.subheader("Performance by Question Difficulty")
                            
                            diff_scores = {}
                            for difficulty, scores in results['difficulty_analysis'].items():
                                diff_scores[difficulty] = scores.reset_index()
                                diff_scores[difficulty].columns = ['Model', f'{difficulty} Score']
                            
                            if diff_scores:
                                # Create separate charts for each difficulty
                                fig, axes = plt.subplots(1, len(diff_scores), figsize=(15, 5))
                                if len(diff_scores) == 1:
                                    axes = [axes]  # Make it iterable if only one difficulty
                                    
                                for i, (difficulty, df) in enumerate(diff_scores.items()):
                                    sns.barplot(x='Model', y=f'{difficulty} Score', data=df, ax=axes[i], palette='viridis')
                                    axes[i].set_title(f'{difficulty} Questions')
                                    axes[i].set_ylim(0, 30)
                                    # Rotate labels if needed
                                    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
                                    
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            # Feature importance analysis
                            st.subheader("Model Strengths by Criteria")
                            
                            feature_names = {'accuracy': 'Accuracy', 'completeness': 'Completeness', 'clarity': 'Clarity'}
                            feature_dfs = []
                            
                            for feature, scores in results['feature_importance'].items():
                                feature_df = scores.reset_index()
                                feature_df.columns = ['Model', feature_names[feature]]
                                feature_dfs.append(feature_df)
                            
                            # Merge all feature dataframes
                            if feature_dfs:
                                feature_analysis = feature_dfs[0]
                                for df in feature_dfs[1:]:
                                    feature_analysis = feature_analysis.merge(df, on='Model')
                                
                                # Create a heatmap
                                plt.figure(figsize=(10, 6))
                                feature_matrix = feature_analysis.set_index('Model')
                                sns.heatmap(feature_matrix, annot=True, cmap='viridis', linewidths=.5, fmt='.1f')
                                plt.title('Model Performance by Criteria')
                                st.pyplot(plt)
                            
                            # Clustering results
                            if results['clustering'] is not None:
                                st.subheader("Model Clustering Analysis")
                                
                                # Display cluster assignments
                                cluster_df = results['clustering'][['model_name', 'cluster']].copy()
                                cluster_df.columns = ['Model', 'Cluster']
                                
                                # Create meaningful cluster labels
                                cluster_means = results['clustering'].groupby('cluster')[['accuracy', 'completeness', 'clarity', 'difficulty_encoded']].mean()
                                cluster_means['total'] = cluster_means.sum(axis=1)
                                cluster_ranks = cluster_means['total'].rank(ascending=False).astype(int)
                                
                                # Map cluster numbers to performance tiers
                                tier_map = {cluster: f"Tier {rank}" for cluster, rank in cluster_ranks.items()}
                                cluster_df['Performance Tier'] = cluster_df['Cluster'].map(tier_map)
                                
                                # Show cluster assignments
                                st.write("Models grouped by performance similarity:")
                                st.dataframe(cluster_df[['Model', 'Performance Tier']], use_container_width=True)
                                
                                # Display PCA visualization if available
                                if results['pca_data'] is not None:
                                    st.subheader("Model Performance Space (PCA)")
                                    
                                    # Create PCA visualization
                                    pca_df = pd.DataFrame(
                                        data=results['pca_data'], 
                                        columns=['Principal Component 1', 'Principal Component 2']
                                    )
                                    pca_df['Model'] = results['pca_models']
                                    pca_df = pca_df.merge(cluster_df, on='Model')
                                    
                                    # Calculate variance explained
                                    variance_explained = results['pca_variance']
                                    
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    sns.scatterplot(
                                        data=pca_df, 
                                        x='Principal Component 1', 
                                        y='Principal Component 2', 
                                        hue='Performance Tier',
                                        s=200,
                                        palette='viridis',
                                        ax=ax
                                    )
                                    
                                    # Add model names as annotations
                                    for idx, row in pca_df.iterrows():
                                        ax.annotate(
                                            row['Model'], 
                                            (row['Principal Component 1'], row['Principal Component 2']),
                                            xytext=(5, 5),
                                            textcoords='offset points',
                                            fontsize=10
                                        )
                                    
                                    # Add variance explained to axis labels
                                    ax.set_xlabel(f"Principal Component 1 ({variance_explained[0]:.2%} variance)")
                                    ax.set_ylabel(f"Principal Component 2 ({variance_explained[1]:.2%} variance)")
                                    ax.set_title("Model Performance Clustering in 2D Space")
                                    
                                    st.pyplot(fig)
                                    
                                    # Explanation
                                    st.info("""
                                    **Interpretation:** 
                                    - Models clustered together have similar performance patterns
                                    - Distance between models indicates how differently they perform
                                    - The axes represent the main factors of variation in model performance
                                    """)
                                    
                            # Conclusion section
                            st.subheader("Conclusion")
                            recommendation = f"""
                            Based on the machine learning analysis of the quiz results, **{best_model}** is the best 
                            performing model for accounting tasks with an average score of {best_score:.2f}/30.
                            """
                            
                            # Check if we have difficulty-specific recommendations
                            best_by_difficulty = {}
                            for difficulty, scores in results['difficulty_analysis'].items():
                                if not scores.empty:
                                    best_by_difficulty[difficulty] = scores.index[0]
                            
                            if len(best_by_difficulty) > 0:
                                recommendation += "\n\n**Best model by difficulty level:**"
                                for difficulty, model in best_by_difficulty.items():
                                    recommendation += f"\n- {difficulty} questions: **{model}**"
                            
                            st.markdown(recommendation)
                    
                    conn.close()
                    
                except Exception as e:
                    st.error(f"Error during machine learning analysis: {str(e)}")
        
        # Documentation for the ML analysis
        with st.expander("How the Analysis Works"):
            st.markdown("""
            ### Machine Learning Methodology
            
            This analysis uses several machine learning techniques to determine the best model for accounting tasks:
            
            1. **Basic Statistical Analysis**
               - Average scores by model
               - Performance breakdown by question difficulty
               - Performance by evaluation criteria (accuracy, completeness, clarity)
            
            2. **Clustering (K-Means)**
               - Groups models by similar performance patterns
               - Helps identify performance tiers
            
            3. **Dimensionality Reduction (PCA)**
               - Visualizes model performance in 2D space
               - Shows relationships between models
            
            ### Data Used
            
            The analysis uses all quiz responses with evaluation data, including:
            - Model accuracy scores
            - Model completeness scores
            - Model clarity scores 
            - Question difficulty
            
            ### Minimum Requirements
            
            At least 10 evaluated responses are needed for meaningful analysis.
            """)
        
    # Return to quiz mode info
    st.markdown("---")
    st.info("To run a benchmark, select models in the sidebar and click 'Start Benchmark'.")
elif len(st.session_state.models_to_test) > 0:
    # Display generated questions first
    if st.session_state.questions_generated and not st.session_state.quiz_complete:
        st.header("Accounting Questions")
        
        # Display the 3 questions with difficulty levels
        for i, question in enumerate(st.session_state.accounting_questions):
            with st.container(border=True):
                st.subheader(f"{question['difficulty']} Question")
                st.markdown(f"**Q{i+1}:** {question['question']}")
        
        # Process questions one by one
        if st.session_state.processing:
            st.header("Processing Questions")
            
            total_questions = len(st.session_state.accounting_questions)
            total_models = len(st.session_state.models_to_test)
            
            # Calculate overall progress percentage
            total_combinations = total_questions * total_models
            current_combination = (st.session_state.current_question_index * total_models) + st.session_state.current_model_index
            overall_progress = current_combination / total_combinations
            
            # Show overall progress
            st.progress(overall_progress)
            
            # Show current status
            current_question = st.session_state.accounting_questions[st.session_state.current_question_index]
            current_model = st.session_state.models_to_test[st.session_state.current_model_index]
            
            st.info(f"Processing {current_question['difficulty']} Question ({st.session_state.current_question_index+1}/{total_questions}) with model: **{current_model}**")
            
            # Process current question/model combination
            q_idx = st.session_state.current_question_index
            model = current_model
            
            # Initialize response dictionary if needed
            if q_idx not in st.session_state.responses:
                st.session_state.responses[q_idx] = {}
            
            # Get response for current question/model
            response_data = query_ollama(model, current_question['question'], 
                                   "You are an expert accountant. Provide a detailed and accurate answer to the accounting question.")
            
            st.session_state.responses[q_idx][model] = {
                "response": response_data["response"],
                "response_time": response_data["response_time"],
                "prompt_tokens": response_data["prompt_tokens"],
                "completion_tokens": response_data["completion_tokens"],
                "total_tokens": response_data["total_tokens"],
                "evaluation": None
            }
            
            # Evaluate the response
            if not response_data["response"].startswith("Error:"):
                evaluation = evaluate_response(
                    current_question['question'],
                    current_question['answer'],
                    response_data,
                    st.session_state.evaluator_model
                )
            else:
                evaluation = {
                    "accuracy": 0,
                    "completeness": 0,
                    "clarity": 0,
                    "total_score": 0,
                    "feedback": "Response contained an error and could not be evaluated.",
                    "response_time": response_data["response_time"],
                    "prompt_tokens": response_data["prompt_tokens"],
                    "completion_tokens": response_data["completion_tokens"],
                    "total_tokens": response_data["total_tokens"]
                }
            
            # We can't calculate final_score yet because we need min/max values across all responses
            # This will be calculated later in the results display
            
            st.session_state.responses[q_idx][model]["evaluation"] = evaluation
            st.session_state.scores[model] += evaluation.get("total_score", 0)
            
            # Save to database
            save_response(
                st.session_state.quiz_id,
                current_question['question'],
                model,
                response_data["response"],
                evaluation,
                st.session_state.evaluator_model
            )
            
            # Update indices for next iteration
            st.session_state.current_model_index += 1
            if st.session_state.current_model_index >= total_models:
                st.session_state.current_model_index = 0
                st.session_state.current_question_index += 1
            
            # Check if we're done processing all questions
            if st.session_state.current_question_index >= total_questions:
                st.session_state.processing = False
                st.session_state.quiz_complete = True
            
            # Rerun to show progress or complete
            st.rerun()
    
    # Show progress while processing
    if st.session_state.processing and not st.session_state.questions_generated:
        st.info("Generating questions... Please wait.")
        st.progress(0.1)  # Show initial progress
    
    # Display final results
    elif st.session_state.quiz_complete:
        st.header("Quiz Results")
        
        # Create aggregate model scores using both scoring systems
        model_scores = {}
        model_final_scores = {}
        
        # First collect all responses to find min/max values for normalization
        all_response_times = []
        all_token_counts = []
        
        for q_idx, question in enumerate(st.session_state.accounting_questions):
            for model in st.session_state.models_to_test:
                if q_idx in st.session_state.responses and model in st.session_state.responses[q_idx]:
                    response = st.session_state.responses[q_idx][model]["response"]
                    if response.startswith("Error:"):
                        continue
                    
                    eval_data = st.session_state.responses[q_idx][model]["evaluation"]
                    if eval_data:
                        all_response_times.append(eval_data.get("response_time", 0))
                        all_token_counts.append(eval_data.get("total_tokens", 0))
                        
                        # Initialize counters if needed
                        if model not in model_scores:
                            model_scores[model] = 0
                            model_final_scores[model] = 0

                        # Add to traditional score
                        model_scores[model] += eval_data.get("total_score", 0)
        
        # Determine min/max values for normalization
        min_response_time = min(all_response_times) if all_response_times else 0
        max_response_time = max(all_response_times) if all_response_times else 1
        min_tokens = min(all_token_counts) if all_token_counts else 0
        max_tokens = max(all_token_counts) if all_token_counts else 1
        
        # Ensure no division by zero
        if min_response_time == max_response_time:
            max_response_time = min_response_time + 1
        if min_tokens == max_tokens:
            max_tokens = min_tokens + 1
        
        # Calculate final scores
        for q_idx, question in enumerate(st.session_state.accounting_questions):
            for model in st.session_state.models_to_test:
                if q_idx in st.session_state.responses and model in st.session_state.responses[q_idx]:
                    response = st.session_state.responses[q_idx][model]["response"]
                    if response.startswith("Error:"):
                        continue
                    
                    eval_data = st.session_state.responses[q_idx][model]["evaluation"]
                    if eval_data:
                        # Normalize metrics according to formula
                        accuracy_norm = eval_data.get("accuracy", 0) / 10
                        completeness_norm = eval_data.get("completeness", 0) / 10
                        clarity_norm = eval_data.get("clarity", 0) / 10
                        
                        response_time = eval_data.get("response_time", 0)
                        response_time_norm = (max_response_time - response_time) / (max_response_time - min_response_time)
                        
                        tokens_used = eval_data.get("total_tokens", 0)
                        token_efficiency_norm = (max_tokens - tokens_used) / (max_tokens - min_tokens)
                        
                        # Calculate final weighted score
                        final_score = (
                            (0.50 * accuracy_norm) +
                            (0.20 * completeness_norm) +
                            (0.15 * clarity_norm) +
                            (0.10 * response_time_norm) +
                            (0.05 * token_efficiency_norm)
                        )
                        
                        # Round to 3 decimal places
                        final_score = round(final_score, 3)
                        
                        # Add to model's total final score
                        model_final_scores[model] += final_score
                        
                        # Store final_score in the evaluation data
                        eval_data["final_score"] = final_score
                        st.session_state.responses[q_idx][model]["evaluation"] = eval_data
                        
                        # Update final_score in the database
                        try:
                            conn = sqlite3.connect('db/quiz_results.db')
                            c = conn.cursor()
                            c.execute("""
                                UPDATE responses 
                                SET final_score = ? 
                                WHERE quiz_id = ? AND question_id = ? AND model_name = ?
                            """, (
                                final_score,
                                st.session_state.quiz_id,
                                question_id_for_text(conn, question["question"]),
                                model
                            ))
                            conn.commit()
                            conn.close()
                        except Exception as e:
                            st.error(f"Error updating final_score in database: {e}")
        
        # Average the final scores by the number of questions
        num_questions = len(st.session_state.accounting_questions)
        for model in model_final_scores:
            model_final_scores[model] = round(model_final_scores[model] / num_questions, 3)
        
        # Let user choose scoring system
        scoring_system = st.radio(
            "Choose scoring system:",
            ["Traditional (Accuracy + Completeness + Clarity)", "Weighted (Including Response Time & Tokens)"],
            horizontal=True
        )
        
        # Create a sorted list of models by the selected score
        if scoring_system == "Traditional (Accuracy + Completeness + Clarity)":
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            score_description = "Total points (max 30 per question)"
        else:
            sorted_models = sorted(model_final_scores.items(), key=lambda x: x[1], reverse=True)
            score_description = "Weighted score (0-1 scale)"
        
        # Display all scores
        st.markdown("### Final Scores")
        st.markdown(f"*{score_description}*")
        
        # Create modern card design for scores
        cols = st.columns(min(3, len(sorted_models)))  # Max 3 cards per row
        
        for i, (model, score) in enumerate(sorted_models):
            col_index = i % len(cols)
            with cols[col_index]:
                # Create a modern card design
                card = st.container(border=True)
                with card:
                    # Model name with larger font
                    st.markdown(f"<h3 style='text-align: center; margin-bottom: 0px;'>{model}</h3>", unsafe_allow_html=True)
                    
                    # Score with even larger font and center-aligned
                    st.markdown(f"<h1 style='text-align: center; margin: 10px 0; color: {'#1f77b4' if i == 0 else '#7fafdf'};'>{score}</h1>", 
                              unsafe_allow_html=True)
                    
                    # Add a trophy for the winner
                    if i == 0:
                        st.markdown("<div style='text-align: center; margin-top: 5px;'>🏆 Winner</div>", unsafe_allow_html=True)
        
        # Display questions and responses
        st.markdown("### Quiz Questions and Responses")
        
        tabs = st.tabs([f"{q['difficulty']} Question" for q in st.session_state.accounting_questions])
        
        for q_idx, tab in enumerate(tabs):
            with tab:
                question = st.session_state.accounting_questions[q_idx]
                st.markdown(f"**Question:** {question['question']}")
                st.markdown("**Correct Answer Concepts:**")
                st.markdown(f"{question['answer']}")
                st.markdown("---")
                
                # Create a tab for each model's response
                if q_idx in st.session_state.responses:
                    model_tabs = st.tabs(st.session_state.models_to_test)
                    
                    for i, model in enumerate(st.session_state.models_to_test):
                        with model_tabs[i]:
                            if model in st.session_state.responses[q_idx]:
                                response = st.session_state.responses[q_idx][model]["response"]
                                if response.startswith("Error:"):
                                    continue
                                
                                st.markdown("### Response:")
                                st.markdown(response)
                                
                                eval_data = st.session_state.responses[q_idx][model]["evaluation"]
                                if eval_data:
                                    st.markdown("### Evaluation:")
                                    col1, col2, col3, col4 = st.columns(4)
                                    col1.metric("Accuracy", eval_data.get("accuracy", "N/A"))
                                    col2.metric("Completeness", eval_data.get("completeness", "N/A"))
                                    col3.metric("Clarity", eval_data.get("clarity", "N/A"))
                                    col4.metric("Total", eval_data.get("total_score", "N/A"))
                                    
                                    # Add a second row for response time and token count
                                    col1, col2 = st.columns(2)
                                    col1.metric("Response Time", f"{eval_data.get('response_time', 0):.2f} sec")
                                    col2.metric("Total Tokens", eval_data.get("total_tokens", 0))
                                    
                                    # Add a third row for token details
                                    col1, col2 = st.columns(2)
                                    col1.metric("Prompt Tokens", eval_data.get("prompt_tokens", 0))
                                    col2.metric("Completion Tokens", eval_data.get("completion_tokens", 0))
                                    
                                    st.markdown("**Feedback:**")
                                    st.markdown(f"{eval_data.get('feedback', 'No feedback')}")
        
        # Detailed score breakdown
        st.markdown("### Detailed Score Breakdown")
        
        # Prepare data for the summary table
        summary_data = []
        
        # First collect all responses to find min/max values for normalization
        all_response_times = []
        all_token_counts = []
        
        for q_idx, question in enumerate(st.session_state.accounting_questions):
            for model in st.session_state.models_to_test:
                if q_idx in st.session_state.responses and model in st.session_state.responses[q_idx]:
                    response = st.session_state.responses[q_idx][model]["response"]
                    if response.startswith("Error:"):
                        continue
                        
                    eval_data = st.session_state.responses[q_idx][model]["evaluation"]
                    if eval_data:
                        all_response_times.append(eval_data.get("response_time", 0))
                        all_token_counts.append(eval_data.get("total_tokens", 0))
        
        # Determine min/max values for normalization
        min_response_time = min(all_response_times) if all_response_times else 0
        max_response_time = max(all_response_times) if all_response_times else 1
        min_tokens = min(all_token_counts) if all_token_counts else 0
        max_tokens = max(all_token_counts) if all_token_counts else 1
        
        # Ensure no division by zero
        if min_response_time == max_response_time:
            max_response_time = min_response_time + 1
        if min_tokens == max_tokens:
            max_tokens = min_tokens + 1
            
        # Now create the summary data with normalized scores
        for q_idx, question in enumerate(st.session_state.accounting_questions):
            for model in st.session_state.models_to_test:
                if q_idx in st.session_state.responses and model in st.session_state.responses[q_idx]:
                    response = st.session_state.responses[q_idx][model]["response"]
                    if response.startswith("Error:"):
                        continue
                        
                    eval_data = st.session_state.responses[q_idx][model]["evaluation"]
                    if eval_data:
                        # Normalize metrics according to formula
                        accuracy_norm = eval_data.get("accuracy", 0) / 10
                        completeness_norm = eval_data.get("completeness", 0) / 10
                        clarity_norm = eval_data.get("clarity", 0) / 10
                        
                        response_time = eval_data.get("response_time", 0)
                        response_time_norm = (max_response_time - response_time) / (max_response_time - min_response_time)
                        
                        tokens_used = eval_data.get("total_tokens", 0)
                        token_efficiency_norm = (max_tokens - tokens_used) / (max_tokens - min_tokens)
                        
                        # Calculate final weighted score
                        final_score = (
                            (0.50 * accuracy_norm) +
                            (0.20 * completeness_norm) +
                            (0.15 * clarity_norm) +
                            (0.10 * response_time_norm) +
                            (0.05 * token_efficiency_norm)
                        )
                        
                        # Round to 3 decimal places
                        final_score = round(final_score, 3)
                        
                        summary_data.append({
                            'Question': f"{question['difficulty']}: {question['question']}",
                            'Model': model,
                            'Accuracy': eval_data.get("accuracy", 0),
                            'Completeness': eval_data.get("completeness", 0),
                            'Clarity': eval_data.get("clarity", 0),
                            'Total': eval_data.get("total_score", 0),
                            'Response Time (s)': round(eval_data.get("response_time", 0), 2),
                            'Prompt Tokens': eval_data.get("prompt_tokens", 0),
                            'Completion Tokens': eval_data.get("completion_tokens", 0),
                            'Total Tokens': eval_data.get("total_tokens", 0),
                            'Final Score': final_score
                        })
        
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            # Hide index column and show full question
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Create pivot table for better visualization
            pivot_df = summary_df.pivot_table(
                index='Question', 
                columns='Model', 
                values='Total',
                aggfunc='sum'
            )
            # Reset index to make the dataframe compatible with st.bar_chart
            pivot_df = pivot_df.reset_index()
            # Use a regular index instead of the Question text
            pivot_df.index = range(len(pivot_df))
        
        # Reset button
        if st.button("Start New Benchmark"):
            st.session_state.scores = {}
            st.session_state.responses = {}
            st.session_state.quiz_complete = False
            st.session_state.models_to_test = []
            st.session_state.processing = False
            st.session_state.quiz_id = None
            st.session_state.accounting_questions = []
            st.session_state.current_question_index = 0
            st.session_state.current_model_index = 0
            st.session_state.questions_generated = False
            st.rerun()
    else:
        st.info("Starting quiz... Please wait.")
else:
    # Default view for non-admin users or before quiz starts
    st.info("👈 Please select models to test in the sidebar and click 'Start Benchmark'")
    
    # If user is not logged in as admin, show a message about admin features
    if not st.session_state.admin_logged_in:
        st.markdown("---")
        st.markdown("**Admin users:** Log in through the sidebar to access the database review interface.")
