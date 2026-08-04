import json
from query import get_rag_chain_and_retriever, ask_question

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE = os.path.join(BASE_DIR, "..", "rag_assignment", "questions_eval.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "rag_assignment", "answers.jsonl")

def main():
    print(f"Loading questions from {EVAL_FILE}...")
    try:
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Could not find {EVAL_FILE}.")
        return

    questions = []
    for line in lines:
        if line.strip():
            questions.append(json.loads(line))

    print("Initializing RAG chain...")
    chain_tuple = get_rag_chain_and_retriever()
    
    results = []
    
    for q_obj in questions:
        q_id = q_obj.get("question_id", "Unknown")
        question_text = q_obj.get("question", "")
        
        try:
            print(f"\nProcessing [{q_id}]: {question_text}")
        except UnicodeEncodeError:
            print(f"\nProcessing [{q_id}]: {question_text.encode('ascii', 'replace').decode('ascii')}")
        
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                answer, sources = ask_question(question_text, chain_tuple)
                break
            except Exception as e:
                print(f"        -> API Error (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print("        -> Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print("        -> Failed after 3 retries. Skipping question.")
                    answer = "I CANNOT ANSWER"
                    sources = []
                    
        time.sleep(3) # Short delay between questions to avoid hitting rate limits
        
        # Check if the system couldn't answer
        if answer == "I CANNOT ANSWER":
            answered = False
            final_answer = ""
            final_sources = []
        else:
            answered = True
            final_answer = answer
            # The assignment asks for a specific format for sources, dropping duplicates
            unique_sources = []
            seen = set()
            for s in sources:
                key = f"{s['file']}_{s['page']}"
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(s)
            final_sources = unique_sources
            
        result_obj = {
            "question_id": q_id,
            "answered": answered,
            "answer": final_answer,
            "sources": final_sources
        }
        results.append(result_obj)
        
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
