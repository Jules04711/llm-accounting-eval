import streamlit as st
import requests
import json
import sqlite3
import os
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Accounting Knowledge Quiz", page_icon="💼", layout="wide")

# Accounting questions
accounting_questions = [
    {
        "difficulty": "Easy",
        "question": "What is the accounting equation and explain its components?",
        "answer": "The accounting equation is Assets = Liabilities + Equity. Assets are resources owned by a business. Liabilities are obligations or debts owed to others. Equity represents the owner's interest in the business."
    },
    {
        "difficulty": "Easy",
        "question": "Explain the difference between accrual accounting and cash accounting.",
        "answer": "Accrual accounting records revenues when earned and expenses when incurred, regardless of when cash changes hands. Cash accounting records transactions only when cash is received or paid. Accrual accounting provides a more accurate picture of financial position but is more complex."
    },
    {
        "difficulty": "Intermediate",
        "question": "Explain the concept of depreciation and the different methods of calculating it.",
        "answer": "Depreciation is the systematic allocation of an asset's cost over its useful life. Methods include: Straight-line (equal amounts each period), Declining balance (accelerated depreciation with higher amounts in earlier years), Units of production (based on actual usage), and Sum-of-the-years'-digits (accelerated method based on remaining useful life)."
    },
    {
        "difficulty": "Intermediate",
        "question": "What is the purpose of a bank reconciliation statement and how is it prepared?",
        "answer": "A bank reconciliation statement reconciles a company's bank account balance with its accounting records. It identifies discrepancies by accounting for outstanding checks, deposits in transit, bank fees, errors, and other items. Preparation involves comparing the bank statement with the company's ledger and adjusting for timing differences and errors."
    },
    {
        "difficulty": "Advanced",
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
    
    # Insert questions if needed
    for q in accounting_questions:
        c.execute("SELECT * FROM questions WHERE question_text = ?", (q["question"],))
        if not c.fetchone():
            c.execute("INSERT INTO questions (difficulty, question_text, correct_answer) VALUES (?, ?, ?)",
                     (q["difficulty"], q["question"], q["answer"]))
    
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
        
        # Get question ID
        c.execute("SELECT question_id FROM questions WHERE question_text = ?", (question_text,))
        question_id = c.fetchone()[0]
        
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
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
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
        st.session_state.current_question = 0
        st.session_state.scores = {model: 0 for model in st.session_state.models_to_test}
        st.session_state.responses = {}
        st.session_state.quiz_complete = False
        st.session_state.processing = False
        st.session_state.quiz_id = create_new_quiz(evaluator_model)
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
    # Progress bar
    total_questions = len(accounting_questions)
    current_question = st.session_state.current_question + 1
    progress_percentage = current_question / total_questions
    
    st.markdown(f"### Progress: Question {current_question}/{total_questions}")
    st.progress(progress_percentage)
    
    # Display current question
    if not st.session_state.quiz_complete:
        current_q = accounting_questions[st.session_state.current_question]
        st.subheader(f"Question {st.session_state.current_question + 1}: {current_q['difficulty']}")
        st.markdown(f"**{current_q['question']}**")
        
        # Process responses for current question automatically
        has_responses = (st.session_state.current_question in st.session_state.responses and 
                        len(st.session_state.responses[st.session_state.current_question]) == len(st.session_state.models_to_test))
        
        if not has_responses and not st.session_state.processing:
            st.session_state.processing = True
            
            with st.spinner("Getting responses from models..."):
                # Get responses from all models
                for model in st.session_state.models_to_test:
                    response = query_ollama(model, current_q['question'], 
                                          "You are an expert accountant. Provide a detailed and accurate answer to the accounting question.")
                    
                    if st.session_state.current_question not in st.session_state.responses:
                        st.session_state.responses[st.session_state.current_question] = {}
                    
                    st.session_state.responses[st.session_state.current_question][model] = {
                        "response": response,
                        "evaluation": None
                    }
            
            with st.spinner("Evaluating responses..."):
                # Evaluate responses
                for model in st.session_state.models_to_test:
                    model_response = st.session_state.responses[st.session_state.current_question][model]["response"]
                    
                    if model_response.startswith("Error:"):
                        evaluation = {
                            "accuracy": 0,
                            "completeness": 0,
                            "clarity": 0,
                            "total_score": 0,
                            "feedback": "Response contained an error and could not be evaluated."
                        }
                    else:
                        evaluation = evaluate_response(
                            current_q['question'],
                            current_q['answer'],
                            model_response,
                            st.session_state.evaluator_model
                        )
                    
                    st.session_state.responses[st.session_state.current_question][model]["evaluation"] = evaluation
                    st.session_state.scores[model] += evaluation.get("total_score", 0)
                    
                    # Save to database
                    save_response(
                        st.session_state.quiz_id,
                        current_q['question'],
                        model,
                        model_response,
                        evaluation,
                        st.session_state.evaluator_model
                    )
            
            st.session_state.processing = False
            st.rerun()
        
        # Display current question responses
        if has_responses:
            st.markdown("### Current Question Results")
            for model in st.session_state.models_to_test:
                response = st.session_state.responses[st.session_state.current_question][model]["response"]
                if response.startswith("Error:"):
                    continue
                    
                with st.expander(f"Response from {model}", expanded=True):
                    st.markdown(response)
                    
                    eval_data = st.session_state.responses[st.session_state.current_question][model]["evaluation"]
                    if eval_data:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", eval_data.get("accuracy", "N/A"))
                        col2.metric("Completeness", eval_data.get("completeness", "N/A"))
                        col3.metric("Clarity", eval_data.get("clarity", "N/A"))
                        col4.metric("Total", eval_data.get("total_score", "N/A"))
                        st.markdown(f"<small>**Feedback:** {eval_data.get('feedback', 'No feedback')}</small>", unsafe_allow_html=True)

            # Navigation buttons
            col1, col2 = st.columns(2)
            if st.session_state.current_question > 0:
                if col1.button("Previous Question"):
                    st.session_state.current_question -= 1
                    st.rerun()
                    
            if st.session_state.current_question < len(accounting_questions) - 1:
                if col2.button("Next Question"):
                    st.session_state.current_question += 1
                    st.rerun()
            else:
                if col2.button("Complete Quiz"):
                    st.session_state.quiz_complete = True
                    st.rerun()
        
        # Display previous questions results
        if st.session_state.current_question > 0:
            st.markdown("### Previous Questions Results")
            for q_idx in range(st.session_state.current_question):
                with st.expander(f"Question {q_idx + 1}: {accounting_questions[q_idx]['question']}"):
                    for model in st.session_state.models_to_test:
                        if q_idx in st.session_state.responses and model in st.session_state.responses[q_idx]:
                            response = st.session_state.responses[q_idx][model]["response"]
                            if response.startswith("Error:"):
                                continue
                                
                            st.markdown(f"**{model}**")
                            st.markdown(response)
                            
                            eval_data = st.session_state.responses[q_idx][model]["evaluation"]
                            if eval_data:
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Accuracy", eval_data.get("accuracy", "N/A"))
                                col2.metric("Completeness", eval_data.get("completeness", "N/A"))
                                col3.metric("Clarity", eval_data.get("clarity", "N/A"))
                                col4.metric("Total", eval_data.get("total_score", "N/A"))
                                st.markdown(f"<small>**Feedback:** {eval_data.get('feedback', 'No feedback')}</small>", unsafe_allow_html=True)
                            st.divider()
    
    # Display final results
    if st.session_state.quiz_complete:
        st.header("🏆 Quiz Complete - Final Results")
        
        # Create a sorted list of models by score
        sorted_models = sorted(st.session_state.scores.items(), key=lambda x: x[1], reverse=True)
        
        # Display winner
        st.subheader(f"🥇 Winner: {sorted_models[0][0]} with {sorted_models[0][1]} points")
        
        # Display all scores
        st.markdown("### Final Scores")
        scores_df = pd.DataFrame({
            'Model': [model for model, _ in sorted_models],
            'Total Score': [score for _, score in sorted_models]
        })
        st.bar_chart(scores_df.set_index('Model'))
        
        # Create detailed summary table
        st.markdown("### Detailed Score Breakdown")
        
        # Prepare data for the summary table
        summary_data = []
        
        for q_idx, question in enumerate(accounting_questions):
            for model in st.session_state.models_to_test:
                if q_idx in st.session_state.responses and model in st.session_state.responses[q_idx]:
                    response = st.session_state.responses[q_idx][model]["response"]
                    if response.startswith("Error:"):
                        continue
                        
                    eval_data = st.session_state.responses[q_idx][model]["evaluation"]
                    if eval_data:
                        summary_data.append({
                            'Question': f"Q{q_idx+1}: {question['question'][:50]}...",
                            'Model': model,
                            'Accuracy': eval_data.get("accuracy", 0),
                            'Completeness': eval_data.get("completeness", 0),
                            'Clarity': eval_data.get("clarity", 0),
                            'Total': eval_data.get("total_score", 0)
                        })
        
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
            
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
            
            st.markdown("### Score Comparison by Question")
            # Create a mapping of index to question for the axis
            questions = pivot_df['Question'].tolist()
            # Drop the Question column as it's now in the index
            chart_df = pivot_df.drop(columns=['Question'])
            st.bar_chart(chart_df)
        
        # Comprehensive Response Summary
        st.markdown("### All Responses and Evaluations")
        
        tabs = st.tabs([f"Question {i+1}" for i in range(len(accounting_questions))])
        
        for q_idx, tab in enumerate(tabs):
            with tab:
                question = accounting_questions[q_idx]
                st.markdown(f"**{question['question']}**")
                st.markdown(f"*Difficulty: {question['difficulty']}*")
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
        
        # Reset button
        if st.button("Start New Quiz"):
            st.session_state.current_question = 0
            st.session_state.scores = {}
            st.session_state.responses = {}
            st.session_state.quiz_complete = False
            st.session_state.models_to_test = []
            st.session_state.processing = False
            st.session_state.quiz_id = None
            st.rerun()
else:
    st.info("👈 Please select models to test in the sidebar and click 'Start Quiz'")