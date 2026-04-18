from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import time
import pickle
import numpy as np
import re
import sqlite3
import json
from datetime import datetime

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("threshold.pkl", "rb") as f:
    THRESHOLD = pickle.load(f)

emp_order     = ["SA", "AG", "NP", "PE", "ST", "NR", "SE", "NO"]
housing_order = ["Owner", "Lease", "Rent"]

app = FastAPI(title="ABC Credit Loan Application API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

QUESTIONS = [
    {"field": "mobile", "question": "Please enter your 10-digit mobile number.", "type": "text"},
    {"field": "email", "question": "Please enter your email address.", "type": "text"},
    {"field": "pan", "question": "Please enter your PAN number (format: ABCDE1234F).", "type": "text"},
    {"field": "birth_year", "question": "Please enter your birth year.", "type": "number"},
    {"field": "pincode", "question": "Please enter your 6-digit pincode.", "type": "text"},
    {"field": "Emp_Level", "question": "Select your employment type.", "type": "choice",
     "options": {"1": ["SA", "Salaried"], "2": ["SE", "Separated / No longer active"], "3": ["AG", "Agriculture / Farming"], "4": ["NR", "MGNREGA / Rural Employment"], "5": ["ST", "Student"], "6": ["NP", "Gig Worker / Non-Permanent"], "7": ["PE", "Pensioner"], "8": ["NO", "Non-earning member"]}},
    {"field": "Education Level", "question": "Select your highest education level.", "type": "choice",
     "options": {"0": "Unknown / Not specified", "1": "Primary education", "2": "Secondary (10th pass)", "3": "Higher Secondary (12th pass)", "4": "Graduate", "5": "Postgraduate / Professional"}},
    {"field": "Product_Cate", "question": "Select the loan product you are applying for.", "type": "choice",
     "options": {"1": "Secured Loans (Home / Property)", "2": "Vehicle Loans (Car / Two-Wheeler)", "3": "Personal / Consumer Loans", "4": "Business / Commercial Loans"}},
    {"field": "Loan_Amt", "question": "Enter the loan amount you need (in Rs, between 1 lac and 1 crore).", "type": "number"},
    {"field": "asset_value", "question": "Enter the approximate value of the asset against which you are taking the loan (in Rs).", "type": "number"},
    {"field": "Housing_Category", "question": "Select your current housing status.", "type": "choice",
     "options": {"1": ["Owner", "Own the property"], "2": ["Lease", "Property on lease"], "3": ["Rent", "Renting the property"]}},
    {"field": "Net_Sal", "question": "Enter your monthly net salary / income (in Rs).", "type": "number"},
    {"field": "Region_Level", "question": "Select the type of area you live in.", "type": "choice",
     "options": {"1": "Metro city (Mumbai, Delhi, Bangalore etc.)", "2": "Tier 1 city", "3": "Tier 2 city", "4": "Tier 3 city", "5": "Semi-urban area", "6": "Rural area", "7": "Remote / tribal area"}},
    {"field": "Existing_Liabilities", "question": "Do you have any existing loans or liabilities?", "type": "choice",
     "options": {"1": ["Y", "Yes"], "2": ["N", "No"]}},
    {"field": "tenure", "question": "Select your preferred loan tenure.", "type": "choice",
     "options": {"1": 12, "2": 24, "3": 36, "4": 48, "5": 60}},
]

def validate(field, value, session_data=None, options=None):
    if field == "mobile":
        if not re.fullmatch(r"[6-9]\d{9}", value):
            return False, "Mobile must be 10 digits starting with 6, 7, 8 or 9."
    elif field == "email":
        if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", value):
            return False, "Please enter a valid email address."
    elif field == "pan":
        if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", value.upper()):
            return False, "PAN format must be ABCDE1234F."
    elif field == "birth_year":
        if not value.isdigit() or not (1966 <= int(value) <= 2008):
            return False, "You must be between 18 and 60 years old to apply."
    elif field == "pincode":
        if not re.fullmatch(r"[1-9]\d{5}", value):
            return False, "Pincode must be 6 digits."
    elif field == "Loan_Amt":
        if not value.isdigit() or not (100000 <= int(value) <= 10000000):
            return False, "Loan amount must be between Rs 1,00,000 and Rs 1,00,00,000."
    elif field == "asset_value":
        if not value.isdigit() or int(value) <= 0:
            return False, "Please enter a valid asset value greater than 0."
        if session_data and int(value) < session_data.get("Loan_Amt", 0):
            return False, f"Asset value cannot be less than loan amount (Rs {session_data.get('Loan_Amt', 0):,})."
    elif field == "Net_Sal":
        if not value.isdigit() or int(value) < 10000:
            return False, "Minimum monthly salary to apply is Rs 10,000."
    elif options:
        if value not in options:
            return False, f"Please select a valid option from {list(options.keys())}."
    return True, None

def parse_value(field, value, options):
    if field in ["birth_year", "Loan_Amt", "asset_value", "Net_Sal"]:
        return int(value)
    elif field in ["Education Level", "Product_Cate", "Region_Level"]:
        return int(value)
    elif field == "pan":
        return value.upper()
    elif options:
        selected = options[value]
        if isinstance(selected, list):
            return selected[0]
        return selected
    return value

def predict(application: dict) -> dict:
    tenure     = application.get("tenure", 36)
    loan_amt   = application["Loan_Amt"]
    net_sal    = application["Net_Sal"]
    approx_emi = loan_amt / tenure

    if approx_emi > 0.5 * net_sal:
        return {
            "decision":            "Decline",
            "probability_approve": 0.0,
            "probability_decline": 100.0,
            "feature_vector":      [],
        }

    emp_enc     = emp_order.index(application["Emp_Level"]) if application["Emp_Level"] in emp_order else len(emp_order)
    housing_enc = housing_order.index(application["Housing_Category"]) if application["Housing_Category"] in housing_order else 0
    liabilities = 1 if application["Existing_Liabilities"] == "Y" else 0

    feature_vector = np.array([[
        application["Education Level"],
        emp_enc,
        application["Product_Cate"],
        application["Loan_Amt"],
        application["LTV_Perc"],
        housing_enc,
        application["Net_Sal"],
        application["Region_Level"],
        application["age"],
        liabilities,
    ]])

    feature_vector_scaled = scaler.transform(feature_vector)
    probability           = model.predict_proba(feature_vector_scaled)[0]
    prediction            = 1 if probability[1] >= THRESHOLD else 0
    decision              = "Approve" if prediction == 1 else "Decline"

    return {
        "decision":            decision,
        "probability_approve": round(probability[1] * 100, 2),
        "probability_decline": round(probability[0] * 100, 2),
        "feature_vector":      feature_vector.tolist()[0],
    }

def setup_database(db_path="abc_credit.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id       TEXT PRIMARY KEY,
            timestamp        TEXT,
            mobile           TEXT,
            email            TEXT,
            pan              TEXT,
            age              INTEGER,
            pincode          TEXT,
            asset_value      INTEGER,
            tenure           INTEGER,
            education        INTEGER,
            employment       TEXT,
            product          INTEGER,
            loan_amt         INTEGER,
            ltv_perc         REAL,
            housing          TEXT,
            net_sal          INTEGER,
            region_level     INTEGER,
            liabilities      TEXT,
            feature_vector   TEXT,
            prob_approve     REAL,
            prob_decline     REAL,
            decision         TEXT,
            session_duration REAL
        )
    """)
    conn.commit()
    conn.close()

def log_session(application: dict, result: dict, duration: float, db_path="abc_credit.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid.uuid4()),
        datetime.now().isoformat(),
        application.get("mobile"),
        application.get("email"),
        application.get("pan"),
        application.get("age"),
        application.get("pincode"),
        application.get("asset_value"),
        application.get("tenure"),
        application.get("Education Level"),
        application.get("Emp_Level"),
        application.get("Product_Cate"),
        application.get("Loan_Amt"),
        application.get("LTV_Perc"),
        application.get("Housing_Category"),
        application.get("Net_Sal"),
        application.get("Region_Level"),
        application.get("Existing_Liabilities"),
        json.dumps(result["feature_vector"]),
        result["probability_approve"],
        result["probability_decline"],
        result["decision"],
        round(duration, 2)
    ))
    conn.commit()
    conn.close()

setup_database()

class AnswerRequest(BaseModel):
    session_id: str
    answer:     str

@app.get("/start")
def start():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"step": 0, "data": {}, "start_time": time.time()}
    first_q = QUESTIONS[0]
    return {
        "session_id": session_id,
        "step":       0,
        "field":      first_q["field"],
        "question":   first_q["question"],
        "type":       first_q["type"],
        "options":    first_q.get("options", None),
        "total":      len(QUESTIONS),
    }

@app.post("/message")
def message(req: AnswerRequest):
    session_id = req.session_id
    answer     = req.answer.strip()

    if session_id not in sessions:
        return {"error": "Invalid or expired session. Please call /start again."}

    session  = sessions[session_id]
    step     = session["step"]
    question = QUESTIONS[step]
    field    = question["field"]
    options  = question.get("options", None)

    valid, error = validate(field, answer, session["data"], options)
    if not valid:
        return {
            "session_id": session_id,
            "step":       step,
            "field":      field,
            "question":   question["question"],
            "type":       question["type"],
            "options":    options,
            "error":      error,
        }

    session["data"][field] = parse_value(field, answer, options)

    if field == "asset_value":
        loan_amt = session["data"].get("Loan_Amt", 0)
        session["data"]["LTV_Perc"] = round((loan_amt / session["data"]["asset_value"]) * 100, 2)

    if field == "birth_year":
        session["data"]["age"] = 2026 - int(answer)

    session["step"] += 1

    if session["step"] >= len(QUESTIONS):
        return {
            "session_id": session_id,
            "status":     "complete",
            "message":    "All questions answered. Please call /decision to get your result.",
        }

    next_q = QUESTIONS[session["step"]]
    return {
        "session_id": session_id,
        "step":       session["step"],
        "field":      next_q["field"],
        "question":   next_q["question"],
        "type":       next_q["type"],
        "options":    next_q.get("options", None),
        "error":      None,
        "total":      len(QUESTIONS),
    }

@app.post("/decision")
def decision(req: AnswerRequest):
    session_id = req.session_id

    if session_id not in sessions:
        return {"error": "Invalid or expired session."}

    session     = sessions[session_id]
    application = session["data"]
    duration    = time.time() - session["start_time"]

    result = predict(application)
    log_session(application=application, result=result, duration=duration)
    del sessions[session_id]

    return {
        "session_id":          session_id,
        "decision":            result["decision"],
        "probability_approve": result["probability_approve"],
        "probability_decline": result["probability_decline"],
        "session_duration":    round(duration, 2),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
