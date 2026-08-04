# Hospital Readmission Risk Predictor — Knowledge Base

## About This Tool
This system predicts whether a diabetic patient is likely to be readmitted to hospital within 30 days of discharge. It was trained on 100,000+ real patient records from 130 US hospitals (1999-2008), sourced from the UCI Diabetes 130-US Hospitals dataset. The model is a Logistic Regression classifier, chosen over Random Forest and Gradient Boosting because it gave better recall (0.52) and is interpretable — doctors can see which factors drove a prediction, which matters for clinical trust and regulatory compliance.

This tool is a clinical decision SUPPORT system, not a diagnostic tool. It should never replace professional medical judgment. It flags risk; it does not confirm outcomes.

## Model Performance
- AUC-ROC: 0.591
- Recall (class 1, readmitted): 0.52 — the model catches about 52% of patients who will actually be readmitted within 30 days
- Precision (class 1): 0.11 — of patients flagged as high risk, about 11% are true readmissions
- Why recall is prioritized over precision: missing a high-risk patient (false negative) can mean a patient is sent home and deteriorates. A false alarm (false positive) just means extra monitoring. In healthcare, the cost of missing risk is much higher than the cost of a false alarm.
- Accuracy is intentionally NOT the headline metric. Because only 8.9% of patients are actually readmitted within 30 days, a model that predicts "not readmitted" for everyone would score 91% accuracy while being clinically useless. This is called the "accuracy paradox."

## The 27 Features Used By The Model
1. num_lab_procedures — number of lab tests performed during the encounter. Higher counts often reflect more complex diagnostic workups.
2. num_medications — number of distinct medications administered during the hospital stay. Correlates with severity of illness.
3. time_in_hospital — length of stay in days. Longer stays generally indicate more severe or complicated cases.
4. age — patient's age (midpoint of the original age bracket, e.g. [70-80) becomes 75).
5. num_procedures — number of procedures (not including lab tests) performed during the stay.
6. number_diagnoses — number of distinct diagnoses recorded for the patient during this encounter.
7. total_meds_prescribed — count of diabetes-related medications (out of 23 tracked) that were actively prescribed (not "No").
8. gender — patient's recorded gender.
9. A1Cresult — result of the Hemoglobin A1C test, which measures average blood glucose over the past 2-3 months. Categories: NotTested, Norm (normal), >7, >8. Values above 8 indicate poor long-term glucose control and are associated with higher complication and readmission risk. "NotTested" is itself a risk signal — it can mean the diabetes wasn't being actively monitored.
10. number_inpatient — number of prior inpatient (hospital admission) visits in the past year. A strong predictor: patients with prior inpatient stays are at meaningfully higher risk of readmission.
11. number_outpatient — number of prior outpatient visits in the past year.
12. diag_3_Other, medical_specialty_Unknown, diag_2_Other, race_Caucasian, admission_source_id_7, change, diag_3_Diabetes, diag_1_Other, admission_type_id_2, num_med_increased, diag_2_Diabetes, medical_specialty_InternalMedicine, admission_type_id_3, number_emergency, num_med_decreased, discharge_disposition_id_6 — various encoded categorical signals: diagnosis category (e.g., whether a diagnosis was diabetes-related vs. other), which medical specialty treated the patient, how the patient was admitted, whether medication dosages changed during the stay, and prior emergency visit history.

## What Drives Risk Most (from Random Forest feature importance analysis)
The top clinical drivers of readmission risk, in order of importance, are: num_lab_procedures, num_medications, time_in_hospital, age, and num_procedures. Notably, num_lab_procedures had very low simple correlation with the outcome (0.03) but was the single most important feature overall — this is because feature importance captures non-linear patterns and interactions that simple correlation misses. Prior inpatient visits (number_inpatient) and prior emergency visits (number_emergency) also showed the largest gap in readmission rate between readmitted and non-readmitted patients — patients with a history of hospitalization are meaningfully more likely to return.

## Why a Patient Might Be Flagged HIGH RISK
A patient tends to be flagged HIGH RISK when several of these are true at once: elevated A1C (>7 or >8, or NotTested), a history of prior inpatient or emergency visits, a longer time_in_hospital, a high number of medications and lab procedures (suggesting medical complexity), and a high number_diagnoses. No single feature alone determines the outcome — the model combines all 27 features into one probability score. The output is a probability, not a certainty: "HIGH risk" typically means the model's predicted probability of 30-day readmission crossed a set threshold (0.5 by default), not that readmission is guaranteed.

## Clinical Term Glossary
- **A1C (Hemoglobin A1C)**: A blood test reflecting average blood sugar over 2-3 months. Under 5.7% is normal for non-diabetics; for diabetics, >7% suggests suboptimal control, >8% suggests poor control and elevated complication risk.
- **Glucose serum test (max_glu_serum)**: A blood glucose measurement. Categories: NotTested, Norm, >200, >300 mg/dL. Levels above 200 indicate uncontrolled hyperglycemia.
- **Inpatient visit**: A hospital admission where the patient stays overnight or longer.
- **Outpatient visit**: A visit where the patient is treated and discharged same-day (e.g., clinic visit).
- **Emergency visit**: A visit through the emergency department.
- **Discharge disposition**: Where the patient goes after leaving the hospital (e.g., home, transferred to another facility, skilled nursing facility).
- **Readmission (30-day)**: The patient is admitted to hospital again within 30 days of a previous discharge — a key quality/cost metric hospitals are penalized for in the US (via CMS readmission reduction programs).
- **Class imbalance**: In this dataset, only 8.9% of patients were actually readmitted within 30 days, vs 91.1% who were not. This imbalance required special handling (SMOTE oversampling on the training set) so the model doesn't just learn to always predict "not readmitted."

## Limitations & Honest Caveats
- The model's AUC (0.591) is only modestly better than random guessing (0.5) — it is a screening aid, not a precise diagnostic tool.
- Precision is low (0.11), meaning most patients flagged HIGH RISK will NOT actually be readmitted. This is a deliberate tradeoff to avoid missing true high-risk patients, but doctors should treat a HIGH flag as "worth a closer look," not "will definitely happen."
- Trained on 1999-2008 US hospital data; clinical practices and patient populations have changed since then, so predictions should be interpreted with that context in mind.
- The model does not have access to any information outside its 27 features — it cannot account for social factors, mental health, home support, or anything not captured in the original dataset.