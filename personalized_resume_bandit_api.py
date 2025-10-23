"""
Personalized Resume Bandit API (Single Endpoint)
Author: Arihant B. Angolkar

Description:
A FastAPI microservice that uses contextual bandit logic + Gemini for personalized
coding question generation, rationale scoring, and adaptive learning.
"""

import os
import random
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
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
# 📘 Prepare Dataset & Models
# --------------------------------------------------------------------
def prepare_models():
    data = [
        ["Bachelor", 0, 1, "Embedded Systems", "C,Matlab", "None", "Beginner", "Basic Programming"],
        ["Bachelor", 2, 3, "Web Development", "HTML,CSS,JS", "None", "Intermediate", "Web App Challenge"],
        ["Master", 4, 5, "AI", "Python,ML", "TensorFlow Certified", "Intermediate", "Machine Learning Problem"],
        ["PhD", 6, 8, "AI", "Python,Deep Learning", "TensorFlow Certified", "Advanced", "Research Paper Replication"],
        ["Bachelor", 1, 2, "Cloud Computing", "AWS,Docker", "AWS Certified", "Intermediate", "Cloud Deployment Challenge"],
        ["Master", 3, 6, "Data Science", "Python,Pandas,Numpy", "None", "Advanced", "Data Analysis Task"],
        ["Diploma", 0, 1, "Embedded Systems", "C,Arduino", "None", "Beginner", "Microcontroller Basics"],
        ["Bachelor", 2, 4, "AI", "Python,ML,Data Visualization", "Google AI Certified", "Intermediate", "AI Model Training"],
        ["Bachelor", 1, 3, "Software Engineering", "Java,OOP", "None", "Intermediate", "Object-Oriented Design"],
        ["Master", 5, 7, "Cloud Computing", "AWS,DevOps", "AWS Certified", "Advanced", "System Design Prototype"],
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "Education",
            "Years_of_Experience",
            "Project_Count",
            "Domain",
            "Skills",
            "Certifications",
            "Skill_Level",
            "Question_Type",
        ],
    )

    if not os.path.exists("resume_models.pkl"):
        encoders = {}
        df_enc = df.copy()
        for col in ["Education", "Domain", "Certifications", "Skill_Level", "Question_Type"]:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col])
            encoders[col] = le

        X = df_enc[
            ["Education", "Years_of_Experience", "Project_Count", "Domain", "Certifications", "Skill_Level"]
        ]
        y = df_enc["Question_Type"]

        clf_tree = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X, y)
        ridge_model = Ridge(alpha=1.0).fit(X, np.arange(len(df_enc)))

        joblib.dump((clf_tree, ridge_model, encoders), "resume_models.pkl")
    return df

df = prepare_models()

# --------------------------------------------------------------------
# 🎯 Contextual Bandit
# --------------------------------------------------------------------
class ContextualBandit:
    def __init__(self, n_arms, n_features, epsilon=0.2, lr=0.1, weights_file="resume_bandit.npy"):
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
# 💬 Gemini Question Generator
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
# 🧠 Predict Question Type
# --------------------------------------------------------------------
def predict_question_type(profile):
    clf, ridge, enc = joblib.load("resume_models.pkl")
    df_in = pd.DataFrame([{
        "Education": profile["Education"],
        "Years_of_Experience": profile["Years_of_Experience"],
        "Project_Count": profile["Project_Count"],
        "Domain": profile["Domain"],
        "Certifications": profile["Certifications"],
        "Skill_Level": profile["Skill_Level"],
    }])
    for col in ["Education", "Domain", "Certifications", "Skill_Level"]:
        df_in[col] = enc[col].transform([df_in[col].iloc[0]])
    pred = clf.predict(df_in)[0]
    return enc["Question_Type"].inverse_transform([pred])[0]

# --------------------------------------------------------------------
# ⭐ Reward + Rationale
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
    return round(min(1.0, reward), 2), rationale or ["No specific alignment found."]

# --------------------------------------------------------------------
# 🧩 FastAPI Setup
# --------------------------------------------------------------------
app = FastAPI(
    title="Personalized Resume Bandit API",
    description="FastAPI microservice for AI-driven personalized coding question generation.",
    version="1.0.0"
)

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
# 🚀 SINGLE ENDPOINT — Generate Personalized Question
# --------------------------------------------------------------------
@app.post("/generate_question")
def generate_question(profile: Profile):
    try:
        # 1️⃣ Predict question type
        predicted_label = predict_question_type(profile.dict())

        # 2️⃣ Bandit decision
        question_types = df["Question_Type"].unique()
        bandit = ContextualBandit(len(question_types), 6, epsilon=0.2, lr=0.12)
        mapping = {t: i for i, t in enumerate(question_types)}
        reverse = {i: t for t, i in mapping.items()}

        context = np.array([
            {"Diploma":0,"Bachelor":1,"Master":2,"PhD":3}.get(profile.Education,0),
            profile.Years_of_Experience,
            profile.Project_Count,
            len(profile.Skills),
            {"Beginner":0,"Intermediate":1,"Advanced":2}.get(profile.Skill_Level,1),
            1 if profile.Certifications != "None" else 0
        ], dtype=float)

        arm = bandit.select(context)
        bandit_label = reverse[arm]

        # 3️⃣ Generate question with Gemini
        question = generate_with_gemini(profile.dict(), bandit_label)

        # 4️⃣ Compute reward + rationale
        reward, rationale = simulated_reward(profile.dict(), bandit_label, question)
        bandit.update(arm, context, reward)

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
    return {"message": "✅ Personalized Resume Bandit API is running!"}
