"""
Personalized Resume Bandit API (Single Endpoint)
Author: Arihant B. Angolkar

Description:
FastAPI microservice with online learning:
- Stores new profiles
- Retrains model automatically
- Uses contextual bandit + Gemini for personalized questions
"""

import os
import random
import numpy as np
import pandas as pd
import joblib
import csv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge
import google.generativeai as genai

# --------------------------------------------------------------------
# ⚙️ Gemini Setup
# --------------------------------------------------------------------
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCWbn1w14NezWjhBjGtb3CPO8CE7VjZW0o")
if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set.")
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# --------------------------------------------------------------------
# 📘 Dataset & Model Files
# --------------------------------------------------------------------
BASE_DATA_FILE = "expanded_resume_data.csv"    # pre-filled dataset
DYNAMIC_DATA_FILE = "dynamic_resume_data.csv" # stores new profiles
MODEL_FILE = "resume_models.pkl"
BANDIT_FILE = "resume_bandit.npy"

# --------------------------------------------------------------------
# 🧩 FastAPI Setup
# --------------------------------------------------------------------
app = FastAPI(
    title="Personalized Resume Bandit API",
    description="FastAPI microservice for AI-driven personalized coding question generation with online learning.",
    version="1.1.0"
)

# --------------------------------------------------------------------
# 🔹 Pydantic Profile
# --------------------------------------------------------------------
class Profile(BaseModel):
    Name: str
    Education: str
    Years_of_Experience: int
    Project_Count: int
    Domain: str
    Skills: List[str]
    Certifications: str
    Skill_Level: str
    Fatigue: str = "low"

# --------------------------------------------------------------------
# 🔹 Contextual Bandit
# --------------------------------------------------------------------
class ContextualBandit:
    def __init__(self, n_arms, n_features, epsilon=0.2, lr=0.1, weights_file=BANDIT_FILE):
        self.n_arms = n_arms
        self.n_features = n_features
        self.epsilon = epsilon
        self.lr = lr
        self.weights_file = weights_file
        self.weights = np.load(weights_file) if os.path.exists(weights_file) else np.zeros((n_arms, n_features))

    def select(self, context_vec):
        return random.randrange(self.n_arms) if random.random() < self.epsilon else int(np.argmax(self.weights.dot(context_vec)))

    def update(self, arm, context_vec, reward):
        pred = float(self.weights[arm].dot(context_vec))
        self.weights[arm] += self.lr * (reward - pred) * context_vec
        np.save(self.weights_file, self.weights)
# --------------------------------------------------------------------
# 🔹 Ensure Base Dataset Exists
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# 🔹 Gemini Question Generator
# --------------------------------------------------------------------
def generate_with_gemini(profile, qtype):
    prompt = f"""
Generate one personalized coding or technical question based on this resume:

- Education: {profile['Education']}
- Experience: {profile['Years_of_Experience']} years
- Domain: {profile['Domain']}
- Skills: {', '.join(profile['Skills'])}
- Certifications: {profile['Certifications']}
- Skill Level: {profile['Skill_Level']}

Question Type: {qtype}

Return a clear problem statement and example input/output if relevant.
"""
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return getattr(response, "text", str(response)).strip()

# --------------------------------------------------------------------
# 🔹 Save New Data
# --------------------------------------------------------------------
def save_new_data(profile_dict, question_type):
    file_exists = os.path.exists(DYNAMIC_DATA_FILE)
    with open(DYNAMIC_DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Education","Years_of_Experience","Project_Count","Domain","Skills","Certifications","Skill_Level","Question_Type"])
        writer.writerow([
            profile_dict["Education"],
            profile_dict["Years_of_Experience"],
            profile_dict["Project_Count"],
            profile_dict["Domain"],
            ",".join(profile_dict["Skills"]),
            profile_dict["Certifications"],
            profile_dict["Skill_Level"],
            question_type
        ])

# --------------------------------------------------------------------
# 🔹 Load Full Dataset
# --------------------------------------------------------------------
def load_full_dataset():
    base_df = pd.read_csv(BASE_DATA_FILE)
    if os.path.exists(DYNAMIC_DATA_FILE):
        dynamic_df = pd.read_csv(DYNAMIC_DATA_FILE)
        full_df = pd.concat([base_df, dynamic_df], ignore_index=True)
    else:
        full_df = base_df
    return full_df

# --------------------------------------------------------------------
# 🔹 Train / Retrain Model
# --------------------------------------------------------------------
def train_model():
    df = load_full_dataset()
    encoders = {}
    df_enc = df.copy()

    for col in ["Education","Domain","Certifications","Skill_Level","Question_Type"]:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    X = df_enc[["Education","Years_of_Experience","Project_Count","Domain","Certifications","Skill_Level"]]
    y = df_enc["Question_Type"]

    clf_tree = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X, y)
    ridge_model = Ridge(alpha=1.0).fit(X, np.arange(len(df_enc)))

    joblib.dump((clf_tree, ridge_model, encoders), MODEL_FILE)
    return clf_tree, ridge_model, encoders

def ensure_base_dataset():
    if not os.path.exists(BASE_DATA_FILE):
        sample_data = [
            ["Bachelor", 0, 1, "Embedded Systems", "C,Matlab", "None", "Beginner", "Basic Programming"],
            ["Bachelor", 2, 3, "Web Development", "HTML,CSS,JS", "None", "Intermediate", "Web App Challenge"],
            ["Master", 4, 5, "AI", "Python,ML", "TensorFlow Certified", "Intermediate", "Machine Learning Problem"],
            ["PhD", 6, 8, "AI", "Python,Deep Learning", "TensorFlow Certified", "Advanced", "Research Paper Replication"],
            ["Bachelor", 1, 2, "Cloud Computing", "AWS,Docker", "AWS Certified", "Intermediate", "Cloud Deployment Challenge"]
        ]
        with open(BASE_DATA_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Education","Years_of_Experience","Project_Count","Domain","Skills","Certifications","Skill_Level","Question_Type"])
            writer.writerows(sample_data)
        print(f"✅ Created default base dataset: {BASE_DATA_FILE}")

# --------------------------------------------------------------------
# 🔹 Initialize Base Dataset & Model on Startup
# --------------------------------------------------------------------
ensure_base_dataset()
if not os.path.exists(MODEL_FILE):
        train_model()

# --------------------------------------------------------------------
# 🔹 Predict Question Type
# --------------------------------------------------------------------
def predict_question_type(profile):
    clf, ridge, enc = joblib.load(MODEL_FILE)
    df_in = pd.DataFrame([{
        "Education": profile["Education"],
        "Years_of_Experience": profile["Years_of_Experience"],
        "Project_Count": profile["Project_Count"],
        "Domain": profile["Domain"],
        "Certifications": profile["Certifications"],
        "Skill_Level": profile["Skill_Level"],
    }])
    for col in ["Education","Domain","Certifications","Skill_Level"]:
        # handle unseen labels
        le = enc[col]
        if df_in[col].iloc[0] not in le.classes_:
            le.classes_ = np.append(le.classes_, df_in[col].iloc[0])
        df_in[col] = le.transform([df_in[col].iloc[0]])
    pred = clf.predict(df_in)[0]
    return enc["Question_Type"].inverse_transform([pred])[0]

# --------------------------------------------------------------------
# 🔹 Reward + Rationale
# --------------------------------------------------------------------
def simulated_reward(profile, label, question_text):
    reward, rationale = 0.5, []
    qtext = (question_text or "").lower()
    skills = [s.lower() for s in profile.get("Skills", [])]

    if any(s in qtext for s in skills):
        reward += 0.25
        rationale.append("Question matches candidate skills.")
    if profile["Domain"].lower() in qtext:
        reward += 0.15
        rationale.append("Aligned with candidate domain.")
    if profile["Skill_Level"].lower() == "beginner" and "basic" in qtext:
        reward += 0.1
        rationale.append("Beginner-level question.")
    if profile["Skill_Level"].lower() == "advanced" and "system design" in qtext:
        reward += 0.1
        rationale.append("Advanced-level challenge.")
    return round(min(1.0, reward),2), rationale or ["No specific alignment found."]

# --------------------------------------------------------------------
# 🔹 API Endpoint — Generate Personalized Question
# --------------------------------------------------------------------
@app.post("/generate_question")
def generate_question(profile: Profile, background_tasks: BackgroundTasks):
    try:
        profile_dict = profile.dict()

        # Predict question type using current model
        predicted_label = predict_question_type(profile_dict)

        # Bandit logic
        df_full = load_full_dataset()
        question_types = df_full["Question_Type"].unique()
        bandit = ContextualBandit(len(question_types), 6, epsilon=0.2, lr=0.12)
        mapping = {t:i for i,t in enumerate(question_types)}
        reverse = {i:t for t,i in mapping.items()}

        context = np.array([
            {"Diploma":0,"Bachelor":1,"Master":2,"PhD":3}.get(profile.Education,0),
            profile.Years_of_Experience,
            profile.Project_Count,
            len(profile.Skills),
            {"Beginner":0,"Intermediate":1,"Advanced":2}.get(profile.Skill_Level,1),
            1 if profile.Certifications!="None" else 0
        ], dtype=float)
        arm = bandit.select(context)
        bandit_label = reverse[arm]

        # Generate question
        question = generate_with_gemini(profile_dict, bandit_label)

        # Compute reward + rationale
        reward, rationale = simulated_reward(profile_dict, bandit_label, question)
        bandit.update(arm, context, reward)

        # Save new data + retrain in background
        save_new_data(profile_dict, bandit_label)
        background_tasks.add_task(train_model)

        return {
            "name": profile.Name,
            "predicted_type": predicted_label,
            "bandit_selected_type": bandit_label,
            "question": question,
            "reward": reward,
            "rationale": rationale
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------
# 🌐 Health Check
# --------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "✅ Personalized Resume Bandit API is running with online learning!"}

# --------------------------------------------------------------------
# 🔹 Initialize Model on Startup
# --------------------------------------------------------------------
if not os.path.exists(MODEL_FILE):
    train_model()
