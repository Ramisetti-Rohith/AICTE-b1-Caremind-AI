import streamlit as st
from transformers import pipeline

# 1. Page Title Configuration
st.set_page_config(page_title="CareMind AI", page_icon="🌱", layout="centered")

st.title("🌱 CareMind: Advanced AI Mental Health Portal")
st.write("A safe, private space for students and parents to understand mental well-being.")

# 2. User Role Selection (Student vs Parent Mode)
user_role = st.radio("Please select your profile:", ["Student (For myself)", "Parent (For my child)"])

# 3. Age Input & Classification
age = st.number_input("Enter Age:", min_value=1, max_value=100, value=18)
if age <= 12:
    age_group = "Child"
elif age <= 19:
    age_group = "Adolescent"
else:
    age_group = "Adult"

st.caption(f"Configuring system for: **{user_role}** | Group: **{age_group}**")

# 4. Load Pre-trained AI Sentiment Model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

mood_analyzer = load_model()

# 5. Dynamic Prompt Text based on Selected User Role
prompt_text = "What is on your mind today?" if "Student" in user_role else "Describe your child's behavior or recent changes:"
user_message = st.text_area(prompt_text, placeholder="Type details here...")

# 6. Core Application Processing
if st.button("Consult CareMind AI"):
    if user_message.strip() == "":
        st.warning("Please enter some details first!")
    else:
        # Run text through the AI model
        prediction = mood_analyzer(user_message)[0]
        mood_label = prediction['label']
        confidence = prediction['score']

        st.subheader("📊 AI Well-being Insights:")
        
        # --- NEW NYANCE: NEUTRAL THRESHOLD LOGIC ---
        # If the AI is unsure (confidence is less than 65%), we classify it as Neutral
        if confidence < 0.65:
            st.warning(f"Analysis: Mindset appears Balanced / Neutral (Confidence: {confidence:.2%})")
            st.write("😐 **Current State:** Your mind seems completely steady and balanced right now. No extreme stress or elevated high spirits detected. Keep maintaining this calm, steady rhythm!")
            
        # Scenario A: Clear Negative Sentiment / Stress Detected
        elif mood_label == "NEGATIVE":
            st.error(f"Analysis: Elevated Stress/Anxiety Detected (Confidence: {confidence:.2%})")
            st.info("💡 **Recommended Action Plan:**")
            
            if age_group == "Child":
                st.write("- **For Parents:** Use comforting physical touch, reduce screen time, and engage them in a calming drawing/coloring activity.")
            elif age_group == "Adolescent":
                st.write("- **For Teens/Students:** Try the 4-7-8 breathing technique. Step away from academic studies for a 15-minute walk. Journal your thoughts on paper.")
            else:
                st.write("- **For Adults:** Practice mindfulness grounding techniques. Prioritize a structured sleep schedule and reduce caffeine intake.")
                
            st.markdown("---")
            st.warning("🚨 **Next Stage: Connect with a Human Professional**")
            st.write("If these symptoms persist, we highly recommend consulting a certified specialist. Here are available doctors in our network:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Dr. R.RevathiPathi** *Psychologist (Adolescent Specialist)* Status: 🟢 Online")
                st.button("Request Mock Call", key="dr1")
            with col2:
                st.markdown("**Dr.Ramisetti Rohith** *Clinical Counselor (Family Therapist)* Status: 🟢 Online")
                st.button("Request Mock Appointment", key="dr2")
                
        # Scenario B: Clear Positive Sentiment Detected
        else:
            st.success(f"Analysis: Mindset appears Stable/Positive (Confidence: {confidence:.2%})")
            st.write("🌟 Excellent! Encourage regular gratitude journaling and maintain open channels of communication to keep this healthy momentum going.")

st.markdown("---")
st.caption("⚠️ **Disclaimer:** CareMind is an AI prototype built for an educational internship exercise. It is not a replacement for professional medical diagnosis or psychiatric counseling.")