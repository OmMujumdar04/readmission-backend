"""
Step 5b: Agent logic.
Combines RAG (knowledge base retrieval) with agentic tool-calling
(calling the ML model's /predict logic when asked about a specific patient).
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq
from rag.retrieve import retrieve
from app import run_prediction  # reuse the prediction logic from app.py

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a clinical assistant chatbot for a Hospital Readmission Risk Predictor tool.
You help doctors understand clinical concepts, how the ML model works, and specific patient risk predictions.

Rules:
- For general/clinical/model questions, answer ONLY using the provided CONTEXT. If the context doesn't cover it, say so honestly.
- If the doctor gives specific patient data and asks about their risk, use the predict_readmission_risk tool — do not guess the risk yourself.
- Never claim certainty. This tool estimates risk; it does not diagnose.
- Be concise and clear (3-6 sentences unless a list is clearer).
"""

# Describe the tool to the LLM in the format Groq/OpenAI-style function calling expects
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_readmission_risk",
            "description": "Predicts a specific patient's 30-day hospital readmission risk using the trained ML model. Use this ONLY when the doctor provides concrete patient data and asks for a risk assessment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_lab_procedures": {"type": "number"},
                    "num_medications": {"type": "number"},
                    "time_in_hospital": {"type": "number"},
                    "age": {"type": "number"},
                    "num_procedures": {"type": "number"},
                    "number_diagnoses": {"type": "number"},
                    "total_meds_prescribed": {"type": "number"},
                    "num_med_increased": {"type": "number"},
                    "num_med_decreased": {"type": "number"},
                    "number_inpatient": {"type": "number"},
                    "number_outpatient": {"type": "number"},
                    "number_emergency": {"type": "number"},
                    "gender": {"type": "integer", "description": "1=Male, 0=Female"},
                    "A1Cresult": {"type": "integer", "description": "0=NotTested, 1=Norm, 2=>7, 3=>8"},
                    "change": {"type": "integer", "description": "1=Changed, 0=No change"},
                    "race_Caucasian": {"type": "integer"},
                    "diag_1_Other": {"type": "integer"},
                    "diag_2_Other": {"type": "integer"},
                    "diag_2_Diabetes": {"type": "integer"},
                    "diag_3_Other": {"type": "integer"},
                    "diag_3_Diabetes": {"type": "integer"},
                    "medical_specialty_Unknown": {"type": "integer"},
                    "medical_specialty_InternalMedicine": {"type": "integer"},
                    "admission_source_id_7": {"type": "integer"},
                    "admission_type_id_2": {"type": "integer"},
                    "admission_type_id_3": {"type": "integer"},
                    "discharge_disposition_id_6": {"type": "integer"},
                },
                "required": [
                    "num_lab_procedures", "num_medications", "time_in_hospital", "age",
                    "num_procedures", "number_diagnoses", "total_meds_prescribed",
                    "num_med_increased", "num_med_decreased", "number_inpatient",
                    "number_outpatient", "number_emergency", "gender", "A1Cresult",
                    "change", "race_Caucasian", "diag_1_Other", "diag_2_Other",
                    "diag_2_Diabetes", "diag_3_Other", "diag_3_Diabetes",
                    "medical_specialty_Unknown", "medical_specialty_InternalMedicine",
                    "admission_source_id_7", "admission_type_id_2", "admission_type_id_3",
                    "discharge_disposition_id_6"
                ]
            }
        }
    }
]


def chat_with_agent(user_message: str) -> dict:
    # Always retrieve KB context — cheap, and helps even when a tool call happens too
    retrieved_chunks = retrieve(user_message, top_k=3)
    context_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION:\n{user_message}"}
    ]

    # First call: let the model decide whether it needs the tool
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",  # model decides itself — this is the "agentic" decision point
        temperature=0.2,
        max_tokens=600
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        # No tool needed — plain RAG answer
        return {
            "answer": response_message.content,
            "used_tool": False,
            "sources": [c["text"][:150] + "..." for c in retrieved_chunks]
        }

    # Model decided it needs the prediction tool
    messages.append(response_message)  # record the model's tool request in the conversation

    for tool_call in tool_calls:
        if tool_call.function.name == "predict_readmission_risk":
            args = json.loads(tool_call.function.arguments)
            result = run_prediction(args)  # actually execute the real ML model

            # Feed the tool's result back to the model
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "predict_readmission_risk",
                "content": json.dumps(result)
            })

    # Second call: model writes a final natural-language answer using the tool's result
    final_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=500
    )

    return {
        "answer": final_response.choices[0].message.content,
        "used_tool": True,
        "sources": [c["text"][:150] + "..." for c in retrieved_chunks]
    }