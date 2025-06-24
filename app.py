import streamlit as st
from transformers import pipeline
from pdfminer.high_level import extract_text
from io import StringIO

# Initialize models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_model = pipeline("question-answering")
text_gen = pipeline("text-generation", model="gpt2")

# Extract text
def extract_file_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return StringIO(uploaded_file.read().decode("utf-8")).read()
    return ""

# Summarize text
def summarize_text(text):
    return summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

# Ask Anything
def get_answer(question, context):
    result = qa_model(question=question, context=context)
    return result['answer'], result['score']

# Challenge Questions
def generate_questions(text):
    prompt = f"Generate 3 logical or comprehension questions based on this:\n{text[:1000]}"
    output = text_gen(prompt, max_length=250, num_return_sequences=1)[0]['generated_text']
    return [q.strip() for q in output.split("?") if q.strip()][:3]

# UI Layout
st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📄 Smart Assistant for Research Summarization")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
if uploaded_file:
    raw_text = extract_file_text(uploaded_file)
    st.success("✅ Document uploaded and processed.")
    
    with st.expander("📌 Auto-Generated Summary"):
        summary = summarize_text(raw_text)
        st.write(summary)

    mode = st.radio("Choose Mode:", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        user_q = st.text_input("Ask your question:")
        if user_q:
            answer, score = get_answer(user_q, raw_text)
            st.write("🔍 **Answer:**", answer)
            st.caption(f"Confidence Score: {score:.2f} – Answer is based on uploaded document.")

    elif mode == "Challenge Me":
        st.subheader("🧠 Answer These Questions:")
        questions = generate_questions(raw_text)
        for i, q in enumerate(questions, 1):
            user_ans = st.text_input(f"Q{i}: {q}?")
            if user_ans:
                expected_ans, _ = get_answer(q, raw_text)
                if user_ans.lower().strip() in expected_ans.lower():
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. Expected: {expected_ans}")
