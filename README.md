# CivicDataLab RAG Assignment

A Retrieval-Augmented Generation (RAG) system built to programmatically answer natural language questions over a corpus of Indian gender budget documents.

## Project Structure

- `code/`: Contains the core Python scripts (`ingest.py`, `query.py`, `evaluate.py`), the requirements file, and the ChromaDB vector database.
- `docs/`: Contains the project documentation (`SCOPE.md`, `NOTES.md`, and the original `ASSIGNMENT.md`).
- `rag_assignment/`: Acts strictly as the data folder containing the raw document corpus, JSONL question files, and the final generated `answers.jsonl` output.

## Prerequisites
- Python 3.9+
- An active Nvidia API Key

## Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/Anurag0828/Civic_Data_Labs.git
cd Civic_Data_Labs/code
```

### 2. Environment Setup
Create a virtual environment and install the required dependencies:
```bash
python -m venv .venv

# Activate on Windows:
.\.venv\Scripts\Activate.ps1

# Activate on Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. API Configuration
Create a `.env` file inside the `code/` directory based on the provided `.env.example` and add your key:
```
NVIDIA_API_KEY="your-nvidia-api-key-here"
```

### 4. Running the Pipeline

**Testing Interactively:**
To ask questions interactively in the terminal, run:
```bash
python query.py
```

**Running the Evaluation Script (Automated Pipeline):**
The true power of this system is its automated pipeline. To automatically read the provided questions from `questions_eval.jsonl`, run them through the AI without any manual typing, and generate the final output, run:
```bash
python evaluate.py
```

**Where is the output?**
The generated output file containing all 18 answers and their exact sources will be saved directly to `rag_assignment/answers.jsonl` for easy review!

*(Note: The `evaluate.py` script includes an automatic retry mechanism and safety delays to gracefully handle Nvidia API rate limits and connection resets during generation).*
