# LLM Benchmarking for Accounting

A Streamlit-based application that **benchmarks large language models (LLMs)** on accounting knowledge and evaluates their performance using structured, expert-led criteria.

This tool is designed for **accounting firms, educators, and AI practitioners** who want to assess the clarity, accuracy, and completeness of AI-generated responses to domain-specific prompts—**all in a secure, local environment** using Ollama.

## Overview

This application automatically generates accounting questions of varying difficulty levels, sends them to multiple LLMs via Ollama, and evaluates the responses using quantitative metrics. The results are stored in a SQLite database for analysis and comparison over time.

---

## Features

* **Benchmark multiple LLMs** running via Ollama
* **Evaluate AI model responses** with structured rubrics (accuracy, completeness, clarity)
* **Generate domain-specific accounting questions** with varying difficulty levels (Easy, Medium, Hard)
* **Use a designated evaluator model** to score responses objectively
* **Track and compare model performance** over time using a local SQLite database
* **Visualize results** in dashboards, cards, and performance tables
* **Audit-ready storage** of all model responses and scoring details
* **Token usage analytics** with prompt and completion token tracking
* **Response time measurement** to evaluate model efficiency
* **Weighted scoring system** that calculates final scores based on customizable criteria weights
* **Advanced performance analysis** using machine learning techniques:
  * Performance clustering to identify model tiers
  * Principal Component Analysis (PCA) for visualizing model relationships
  * Difficulty-specific performance scoring
  * Feature importance analysis for model strengths

---

## Requirements

* Python 3.8+
* [Streamlit](https://streamlit.io)
* [Ollama](https://ollama.com) running locally
* SQLite (installed with Python)
* Python packages (installed via requirements.txt):
  * streamlit
  * requests
  * pandas
  * scikit-learn
  * matplotlib
  * seaborn
  * numpy
  * tiktoken

---

## Setup

1. Clone this repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Start the Ollama server locally (default: `http://localhost:11434`)
4. Launch the app:

   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Open the app in your browser (Streamlit will provide a local URL)
2. From the sidebar:
   * Select which LLM models to benchmark
   * Choose an evaluator model (e.g., LLaMA 3 for consistent grading)
3. Click **"Start Benchmark"** to generate accounting questions and initiate model evaluations
4. Review real-time scoring and comparisons
5. Analyze detailed model performance including feedback and score breakdowns

### Benchmark Process

The application works as follows:
1. Generates 3 accounting questions (Easy, Medium, Hard) using the evaluator model
2. Sends each question to all selected LLMs
3. Evaluates responses based on accuracy, completeness, and clarity
4. Stores all data in the SQLite database
5. Presents real-time results and visualizations

### Scoring System

The application implements a comprehensive evaluation framework based on a technical paper on LLM Response Evaluation for Accounting and Auditing. The scoring system assesses responses across five key dimensions:

| Metric | Weight | Description |
|--------|--------|-------------|
| Accuracy | 50% | Correctness of response aligned with GAAP, IFRS, and regulatory requirements |
| Completeness | 20% | Thoroughness in addressing all necessary aspects of an accounting query |
| Clarity | 15% | Intelligibility and straightforwardness of the explanation |
| Response Time | 10% | Speed of the model's response, critical for real-time scenarios |
| Token Efficiency | 5% | Economy of language, influencing costs and readability |

#### Score Normalization

Each dimension is normalized to a [0,1] range to standardize scoring:

* **Accuracy, Completeness, Clarity**: Rated on a scale of 0-10, then normalized by dividing by 10
* **Response Time**: Normalized using a formula that compares to maximum and minimum response times in the dataset (faster responses yield scores closer to 1)
* **Token Efficiency**: Normalized based on maximum and minimum token counts (fewer tokens result in scores closer to 1)

#### Final Score Calculation

The final composite score is calculated using the weighted sum formula:

```
Final Score = (Accuracy × 0.5) + (Completeness × 0.2) + (Clarity × 0.15) + (Response Time × 0.1) + (Token Efficiency × 0.05)
```

This framework prioritizes accuracy while balancing other important factors for practical accounting and auditing applications.

---

## Admin Features

The application includes an admin interface that provides:

* **SQL Database Review**: Run custom queries on the benchmark database
* **ML Model Analysis**: Apply machine learning techniques to identify:
  * Overall best-performing models
  * Best models by difficulty level
  * Performance clusters and tiers
  * Visualization of model relationships
* **Performance metrics**: Advanced scoring including response time and token efficiency

---

## Database Structure

The application uses SQLite with the following schema:

* **quizzes**: Tracks benchmark sessions with timestamp and evaluator model
* **questions**: Stores all generated questions with difficulty level and correct answers
* **responses**: Contains all model responses and evaluation metrics

---

## Ideal Use Cases

* Accounting firms evaluating AI tools for client work
* AI engineers testing model fine-tuning in finance
* Educators comparing LLMs for accounting education
* Audit teams testing LLM compliance in explanation quality
* Researchers analyzing LLM performance patterns in specialized domains

---

## Technical Implementation

* Built with Streamlit for rapid UI development
* Uses sklearn for machine learning analyses
* Implements tiktoken for token counting
* Connects to Ollama API for local LLM inference
* Stores all data in SQLite for persistence and analysis

---

## License

This project is provided as-is. See license file for details.

---

## Contributing

Contributions welcome! Please submit a pull request or open an issue to discuss proposed changes.



