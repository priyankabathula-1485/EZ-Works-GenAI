import streamlit as st
from transformers import pipeline
from pdfminer.high_level import extract_text
from io import StringIO

# Load models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_pipeline = pipeline("question-answering")
generator = pipeline("text-generation", model="gpt2")

def extract_file_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return StringIO(uploaded_file.read().decode("utf-8")).read()
    return ""

st.title("📄 Smart Research Assistant")
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    raw_text = extract_file_text(uploaded_file)
    st.success("Document uploaded successfully.")

    summary = summarizer(raw_text[:2000], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    st.subheader("🔍 Auto-Summary")
    st.write(summary)

    mode = st.radio("Choose a mode", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        user_q = st.text_input("Ask a question from the document:")
        if user_q:
            answer = qa_pipeline(question=user_q, context=raw_text[:2000])
            st.write("Answer:", answer["answer"])
    elif mode == "Challenge Me":
        prompt = f"Generate 3 comprehension questions based on the following:\n{raw_text[:1000]}"
        output = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
        st.subheader("🧠 Challenge Questions")
        for i, q in enumerate(output.split("?")[:3], 1):
            st.write(f"Q{i}: {q.strip()}?")
