# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle
# import pandas as pd
# import numpy as np
# from fastapi.middleware.cors import CORSMiddleware

# # Load model artifacts
# with open('readmission_model.pkl', 'rb') as f:
#     artifacts = pickle.load(f)

# app = FastAPI(title="Hospital Readmission Risk API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class PatientData(BaseModel):
#     # Numerical features
#     num_lab_procedures: float #
#     num_medications: float # 
#     time_in_hospital: float #
#     age: float #
#     num_procedures: float # 
#     number_diagnoses: float #
#     total_meds_prescribed: float # 
#     num_med_increased: float #
#     num_med_decreased: float # 
#     number_inpatient: float # 
#     number_outpatient: float #
#     number_emergency: float #

#     # Binary/encoded features
#     gender: int              # 1=Male, 0=Female  #
#     A1Cresult: int           # 0=NotTested, 1=Norm, 2=>7, 3=>8 #
#     change: int              # 1=Changed, 0=No change # 
#     race_Caucasian: int      # 1=Yes, 0=No #
#     diag_1_Other: int        # 1=Yes, 0=No # 
#     diag_2_Other: int        # 1=Yes, 0=No #
#     diag_2_Diabetes: int     # 1=Yes, 0=No #
#     diag_3_Other: int        # 1=Yes, 0=No #
#     diag_3_Diabetes: int     # 1=Yes, 0=No #
#     medical_specialty_Unknown: int          # 1=Yes, 0=No #
#     medical_specialty_InternalMedicine: int # 1=Yes, 0=No #
#     admission_source_id_7: int  # 1=Yes, 0=No # 
#     admission_type_id_2: int    # 1=Yes, 0=No #
#     admission_type_id_3: int    # 1=Yes, 0=No #
#     discharge_disposition_id_6: int  # 1=Yes, 0=No #

# @app.get("/")
# def home():
#     return {"message": "Hospital Readmission Risk Predictor API 🏥"}

# @app.post("/predict")
# def predict(patient: PatientData):
#     # Step 1: Convert to dataframe
#     patient_dict = patient.model_dump()
#     patient_df = pd.DataFrame([patient_dict])

#     # Step 2: Recreate engineered features
#     patient_df['severity_score'] = patient_df['time_in_hospital'] * patient_df['num_medications']
#     patient_df['condition_complexity'] = patient_df['number_inpatient'] * patient_df['number_diagnoses']

#     # Step 3: Scale numerical features
#     cols_to_scale = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
#                      'num_medications', 'number_outpatient', 'number_emergency',
#                      'number_inpatient', 'number_diagnoses', 'age',
#                      'total_meds_prescribed', 'num_med_increased', 'num_med_decreased',
#                      'severity_score', 'condition_complexity']

#     patient_df[cols_to_scale] = artifacts['scaler'].transform(patient_df[cols_to_scale])

#     # Step 4: Select exactly the features model needs
#     patient_df = patient_df[artifacts['feature_names']]

#     # Step 5: Predict
#     prob = artifacts['model'].predict_proba(patient_df)[0][1]
#     risk = 'HIGH' if prob >= artifacts['threshold'] else 'LOW'

#     return {
#         'readmission_risk': risk,
#         'probability': round(float(prob), 3),
#         'message': f'Patient has {risk} risk of readmission within 30 days'
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)









from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load model artifacts
with open('readmission_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

app = FastAPI(title="Hospital Readmission Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    # Numerical features
    num_lab_procedures: float
    num_medications: float
    time_in_hospital: float
    age: float
    num_procedures: float
    number_diagnoses: float
    total_meds_prescribed: float
    num_med_increased: float
    num_med_decreased: float
    number_inpatient: float
    number_outpatient: float
    number_emergency: float

    # Binary/encoded features
    gender: int              # 1=Male, 0=Female
    A1Cresult: int           # 0=NotTested, 1=Norm, 2=>7, 3=>8
    change: int              # 1=Changed, 0=No change
    race_Caucasian: int      # 1=Yes, 0=No
    diag_1_Other: int        # 1=Yes, 0=No
    diag_2_Other: int        # 1=Yes, 0=No
    diag_2_Diabetes: int     # 1=Yes, 0=No
    diag_3_Other: int        # 1=Yes, 0=No
    diag_3_Diabetes: int     # 1=Yes, 0=No
    medical_specialty_Unknown: int
    medical_specialty_InternalMedicine: int
    admission_source_id_7: int
    admission_type_id_2: int
    admission_type_id_3: int
    discharge_disposition_id_6: int


class ChatMessage(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "Hospital Readmission Risk Predictor API 🏥"}


def run_prediction(patient_dict: dict) -> dict:
    """
    Core prediction logic, reusable by both the /predict endpoint
    and the chatbot's agentic tool call.
    """
    patient_df = pd.DataFrame([patient_dict])

    # Recreate engineered features
    patient_df['severity_score'] = patient_df['time_in_hospital'] * patient_df['num_medications']
    patient_df['condition_complexity'] = patient_df['number_inpatient'] * patient_df['number_diagnoses']

    # Scale numerical features
    cols_to_scale = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                     'num_medications', 'number_outpatient', 'number_emergency',
                     'number_inpatient', 'number_diagnoses', 'age',
                     'total_meds_prescribed', 'num_med_increased', 'num_med_decreased',
                     'severity_score', 'condition_complexity']

    patient_df[cols_to_scale] = artifacts['scaler'].transform(patient_df[cols_to_scale])

    # Select exactly the features model needs
    patient_df = patient_df[artifacts['feature_names']]

    # Predict
    prob = artifacts['model'].predict_proba(patient_df)[0][1]
    risk = 'HIGH' if prob >= artifacts['threshold'] else 'LOW'

    return {
        'readmission_risk': risk,
        'probability': round(float(prob), 3),
        'message': f'Patient has {risk} risk of readmission within 30 days'
    }


@app.post("/predict")
def predict(patient: PatientData):
    return run_prediction(patient.model_dump())


@app.post("/chat")
def chat(chat_message: ChatMessage):
    """
    RAG + agentic chatbot endpoint.
    Imported lazily (inside the function) to avoid a circular import,
    since rag/agent.py itself imports run_prediction from this file.
    """
    from rag.agent import chat_with_agent
    result = chat_with_agent(chat_message.message)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)