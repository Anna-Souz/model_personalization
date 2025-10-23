import os
import streamlit as st
import requests

st.set_page_config(page_title="Resume Bandit Tester", page_icon="🤖", layout="centered")

st.title("🎯 Personalized Resume Bandit – Testing UI")

# ✅ Use environment variable first, fallback to your Render URL
base_url = os.getenv("BACKEND_URL", "https://model-personalization-1.onrender.com/")

# Optional: allow tester to override manually
base_url = st.text_input("Enter FastAPI Base URL", base_url)

st.subheader("🧍 Candidate Profile")
name = st.text_input("Name", "Arihant")
education = st.selectbox("Education", ["Diploma", "Bachelor", "Master", "PhD"])
experience = st.number_input("Years of Experience", 0, 15, 2)
projects = st.number_input("Project Count", 0, 20, 3)
domain = st.selectbox("Domain", ["AI", "Web Development", "Cloud Computing", "Embedded Systems", "Software Engineering"])
skills = st.text_input("Skills (comma separated)", "Python,ML")
certs = st.text_input("Certifications", "Google AI Certified")
level = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"])
fatigue = st.selectbox("Fatigue", ["low", "medium", "high"])

if st.button("🚀 Generate Personalized Question"):
    payload = {
        "Name": name,
        "Education": education,
        "Years_of_Experience": experience,
        "Project_Count": projects,
        "Domain": domain,
        "Skills": [s.strip() for s in skills.split(",")],
        "Certifications": certs,
        "Skill_Level": level,
        "Fatigue": fatigue
    }
    try:
        # Ensure no trailing slash duplication
        res = requests.post(f"{base_url.rstrip('/')}/generate_question", json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            st.success("✅ Question Generated!")
            st.markdown(f"**Predicted Type:** {data['predicted_type']}")
            st.markdown(f"**Bandit Selected Type:** {data['bandit_selected_type']}")
            st.write("### 💡 Question:")
            st.write(data["question"])
            st.write("### 🧠 Rationale:")
            st.write("\n".join(data["rationale"]))
            st.metric("Reward", data["reward"])
        else:
            st.error(f"API Error: {res.text}")
    except Exception as e:
        st.error(f"Connection failed: {e}")
