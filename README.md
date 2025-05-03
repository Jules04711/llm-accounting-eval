# LLM Benchmarking for Accounting

A Streamlit-based application that **benchmarks large language models (LLMs)** on accounting knowledge and evaluates their performance using structured, expert-led criteria.

This tool is designed for **accounting firms, educators, and AI practitioners** who want to assess the clarity, accuracy, and completeness of AI-generated responses to domain-specific prompts—**all in a secure, local environment** using Ollama.

---

## Features

* **Benchmark multiple LLMs** running via Ollama
* **Evaluate AI model responses** with structured rubrics (accuracy, completeness, clarity)
* **Generate domain-specific accounting questions** with varying difficulty levels
* **Use a designated evaluator model** to score responses objectively
* **Track and compare model performance** over time using a local SQLite database
* **Visualize results** in dashboards, cards, and performance tables
* **Audit-ready storage** of all model responses and scoring details

---

## Requirements

* Python 3.8+
* [Streamlit](https://streamlit.io)
* [Ollama](https://ollama.com) running locally
* SQLite (installed with Python)

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
3. Click **“Start Benchmark”** to generate accounting questions and initiate model evaluations
4. Review real-time scoring and comparisons
5. Analyze detailed model performance including feedback and score breakdowns

---

## Ideal Use Cases

* Accounting firms evaluating AI tools for client work
* AI engineers testing model fine-tuning in finance
* Educators comparing LLMs for accounting education
* Audit teams testing LLM compliance in explanation quality
