# LLM Accounting Quiz

A Streamlit application that tests different LLM models on accounting knowledge questions. The app evaluates responses based on accuracy, completeness, and clarity.

## Features
- Test multiple Ollama models simultaneously
- Evaluate model responses with expert criteria
- Track performance across different accounting concepts
- Store quiz results in a local SQLite database
- Generate accounting questions at varying difficulty levels
- Analyze model performance using ML techniques
- Visualize model performance comparisons

## Requirements
- Python 3.8+
- Streamlit
- Ollama running locally
- SQLite

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Run Ollama server locally (must be accessible at http://localhost:11434)
3. Run the app: `streamlit run app.py`

## Usage
1. Select the LLM models you wish to benchmark from the sidebar
2. Choose an evaluator model (preferably a strong model for accurate evaluations)
3. Click "Start Benchmark" to begin the quiz
4. View results in real-time as models respond to accounting questions
5. Review detailed performance analytics after the benchmark completes 
