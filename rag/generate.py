"""
Step 4: Generation function.
Takes retrieved chunks + the user's question, sends them to a free Groq-hosted LLM,
and returns a grounded natural-language answer.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from rag.retrieve import retrieve

load_dotenv()  # reads backend/.env

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Llama 3.3 70B — free on Groq, strong quality, very fast inference
MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a clinical assistant chatbot for a Hospital Readmission Risk Predictor tool.
Your job is to help doctors and staff understand:
1. Clinical concepts (like A1C, glucose tests, readmission)
2. How the machine learning model works and why it makes certain predictions
3. What the model's features and limitations mean

Rules:
- Answer ONLY using the CONTEXT provided below. Do not use outside knowledge.
- If the context does not contain the answer, say clearly: "I don't have enough information in my knowledge base to answer that confidently."
- Be clear and precise. Use plain language, but don't oversimplify to the point of losing clinical accuracy.
- Never state a prediction is certain — this tool provides risk estimates, not diagnoses.
- Keep answers concise (3-6 sentences) unless the question needs a list.
"""


def generate_answer(user_question: str, top_k: int = 3) -> dict:
    # Step A: Retrieve relevant chunks (from Step 3)
    retrieved_chunks = retrieve(user_question, top_k=top_k)
    context_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])

    # Step B: Build the prompt that grounds the LLM in that context
    user_prompt = f"""CONTEXT:
{context_text}

QUESTION:
{user_question}

Answer the question using only the context above."""

    # Step C: Call the LLM
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,  # low temperature = more focused/deterministic, less "creative"
        max_tokens=500
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [c["text"][:150] + "..." for c in retrieved_chunks]  # for transparency in UI
    }


# Quick manual test
if __name__ == "__main__":
    result = generate_answer("What does an A1Cresult of >8 mean clinically?")
    print("ANSWER:\n", result["answer"])
    print("\nSOURCES USED:")
    for s in result["sources"]:
        print(" -", s)