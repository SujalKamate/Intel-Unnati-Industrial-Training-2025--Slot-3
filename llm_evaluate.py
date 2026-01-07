import subprocess
import json

def llm_score(question, answer, context):
    prompt = f"""
Evaluate the answer based on NCERT context.

Question:
{question}

Answer:
{answer}

Context:
{context}

Score each from 1 to 5 in JSON:
{{
  "relevance": int,
  "correctness": int,
  "groundedness": int
}}
"""
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        capture_output=True
    )

    try:
        return json.loads(result.stdout.decode())
    except:
        return None
    

