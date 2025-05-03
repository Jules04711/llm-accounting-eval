import streamlit as st
import requests
import json
import sqlite3
import os
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Accounting Knowledge Quiz", page_icon="💼", layout="wide")

# Function to generate accounting questions using LLM
def generate_accounting_questions(evaluator_model):
    prompt = """
    Generate 3 accounting questions with corresponding detailed answers.
    Questions should be of different difficulty levels:
    1. Easy: Basic accounting concepts
    2. Medium: Intermediate accounting principles 
    3. Hard: Advanced accounting concepts or regulations
    
    Format the output as a JSON array with each question having these fields:
    - difficulty: "Easy", "Medium", or "Hard"
    - question: The accounting question
    - answer: A detailed, comprehensive answer
    
    Return only the JSON array, nothing else.
    """
    
    try:
        response = query_ollama(evaluator_model, prompt)
        
        # Try to extract JSON from the response
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            questions = json.loads(json_str)
            return questions
        else:
            # Fallback questions if JSON parsing fails
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
        FOREIGN KEY (quiz_id) REFERENCES quizzes (quiz_id),
        FOREIGN KEY (question_id) REFERENCES questions (question_id)
    )
    ''')
    
    conn.commit()
    conn.close()

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
        
        # Save response
        c.execute('''
        INSERT INTO responses (
            quiz_id, question_id, model_name, response_text, 
            accuracy, completeness, clarity, total_score, feedback, evaluator_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            evaluator_model
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error saving to database: {e}")

# API function
def query_ollama(model, prompt, system=""):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_response(question, correct_answer, model_response, evaluator_model):
    # Skip evaluation if the response is an error message
    if model_response.startswith("Error:"):
        return {
            "accuracy": 0,
            "completeness": 0,
            "clarity": 0,
            "total_score": 0,
            "feedback": "Response contained an error and could not be evaluated."
        }
        
    evaluation_prompt = f"""
    You are an expert accounting professor evaluating a response to an accounting question.
    
    Question: {question}
    
    Correct answer concepts: {correct_answer}
    
    Model's response: {model_response}
    
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
        json_start = evaluation_result.find('{')
        json_end = evaluation_result.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = evaluation_result[json_start:json_end]
            return json.loads(json_str)
        else:
            return {
                "accuracy": 5,
                "completeness": 5,
                "clarity": 5,
                "total_score": 15,
                "feedback": "Error parsing evaluation"
            }
    except Exception:
        return {
            "accuracy": 5,
            "completeness": 5,
            "clarity": 5,
            "total_score": 15,
            "feedback": "Error during evaluation"
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

# Initialize database
init_db()

# Main app
st.title("🧮 Accounting Knowledge Quiz with Ollama LLM")
st.markdown("This app tests different LLM models on accounting knowledge and determines the winner.")

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
    st.subheader("Select models to test")
    model_selections = {}
    for model in available_models:
        model_selections[model] = st.checkbox(model, value=model in ["llama3", "mistral"] if len(st.session_state.models_to_test) == 0 else model in st.session_state.models_to_test)
    
    # Evaluator model
    evaluator_model = st.selectbox(
        "Select evaluator model",
        available_models,
        index=available_models.index("llama3") if "llama3" in available_models else 0
    )
    
    # Start button
    if st.button("Start Quiz"):
        st.session_state.models_to_test = [model for model, selected in model_selections.items() if selected]
        st.session_state.evaluator_model = evaluator_model
        st.session_state.scores = {model: 0 for model in st.session_state.models_to_test}
        st.session_state.responses = {}
        st.session_state.quiz_complete = False
        st.session_state.processing = True
        st.session_state.quiz_id = create_new_quiz(evaluator_model)
        
        # Generate questions with the evaluator model
        with st.spinner("Generating accounting questions..."):
            st.session_state.accounting_questions = generate_accounting_questions(evaluator_model)
        
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
        
        st.markdown(f"**Total quizzes run:** {num_quizzes}")
        st.markdown(f"**Total model responses:** {num_responses}")
    except Exception:
        st.warning("Database statistics not available.")

# Main content
if len(st.session_state.models_to_test) > 0:
    # Process all quiz questions at once in the background
    if st.session_state.processing and not st.session_state.quiz_complete:
        with st.spinner("Processing all questions... This may take a minute..."):
            # Get responses from all models for all questions
            for q_idx, current_q in enumerate(st.session_state.accounting_questions):
                if q_idx not in st.session_state.responses:
                    st.session_state.responses[q_idx] = {}
                
                for model in st.session_state.models_to_test:
                    response = query_ollama(model, current_q['question'], 
                                          "You are an expert accountant. Provide a detailed and accurate answer to the accounting question.")
                    
                    st.session_state.responses[q_idx][model] = {
                        "response": response,
                        "evaluation": None
                    }
                    
                    # Evaluate the response
                    if not response.startswith("Error:"):
                        evaluation = evaluate_response(
                            current_q['question'],
                            current_q['answer'],
                            response,
                            st.session_state.evaluator_model
                        )
                    else:
                        evaluation = {
                            "accuracy": 0,
                            "completeness": 0,
                            "clarity": 0,
                            "total_score": 0,
                            "feedback": "Response contained an error and could not be evaluated."
                        }
                    
                    st.session_state.responses[q_idx][model]["evaluation"] = evaluation
                    st.session_state.scores[model] += evaluation.get("total_score", 0)
                    
                    # Save to database
                    save_response(
                        st.session_state.quiz_id,
                        current_q['question'],
                        model,
                        response,
                        evaluation,
                        st.session_state.evaluator_model
                    )
            
            st.session_state.processing = False
            st.session_state.quiz_complete = True
            st.rerun()
    
    # Show progress while processing
    if st.session_state.processing:
        st.info("Quiz is running... Please wait while we process all questions.")
        st.progress(0.5)  # Show indeterminate progress
    
    # Display final results
    elif st.session_state.quiz_complete:
        st.header("Quiz Results")
        
        # Create a sorted list of models by score
        sorted_models = sorted(st.session_state.scores.items(), key=lambda x: x[1], reverse=True)
        
        # Display all scores
        st.markdown("### Final Scores")
        
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
                                    st.error(f"Error: This model encountered an error for this question.")
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
                                    
                                    st.markdown("**Feedback:**")
                                    st.markdown(f"{eval_data.get('feedback', 'No feedback')}")
        
        # Detailed score breakdown
        st.markdown("### Detailed Score Breakdown")
        
        # Prepare data for the summary table
        summary_data = []
        
        for q_idx, question in enumerate(st.session_state.accounting_questions):
            for model in st.session_state.models_to_test:
                if q_idx in st.session_state.responses and model in st.session_state.responses[q_idx]:
                    response = st.session_state.responses[q_idx][model]["response"]
                    if response.startswith("Error:"):
                        continue
                        
                    eval_data = st.session_state.responses[q_idx][model]["evaluation"]
                    if eval_data:
                        summary_data.append({
                            'Question': f"{question['difficulty']}: {question['question']}",
                            'Model': model,
                            'Accuracy': eval_data.get("accuracy", 0),
                            'Completeness': eval_data.get("completeness", 0),
                            'Clarity': eval_data.get("clarity", 0),
                            'Total': eval_data.get("total_score", 0)
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
        if st.button("Start New Quiz"):
            st.session_state.scores = {}
            st.session_state.responses = {}
            st.session_state.quiz_complete = False
            st.session_state.models_to_test = []
            st.session_state.processing = False
            st.session_state.quiz_id = None
            st.session_state.accounting_questions = []
            st.rerun()
    else:
        st.info("Starting quiz... Please wait.")
else:
    st.info("👈 Please select models to test in the sidebar and click 'Start Quiz'")
