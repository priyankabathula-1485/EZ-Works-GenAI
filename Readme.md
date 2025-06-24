EZ Works GenAI Assignment – Smart Assistant for Research Summarization

 Objective

Build an AI assistant that:
- Accepts PDF or TXT file uploads
- Generates an auto-summary (≤ 150 words)
- Answers user questions based on the document
- Challenges the user with logic/comprehension questions and evaluates them

 Tech Stack

- Python
- Streamlit (Web UI)
- Hugging Face Transformers (for NLP tasks)
- pdfminer.six (for PDF text extraction)

 Setup Instructions

 1. Clone or Download the Repository
bash
git clone https://github.com/your-username/genai-assistant.git
cd genai-assistant

Create Virtual Environment :-
python -m venv assistant_env
assistant_env\Scripts\activate  # Windows
# OR
source assistant_env/bin/activate  # Mac/Linux

Install Requirements :- 
pip install -r requirements.txt

Run the Application :-
streamlit run app.py

Folder Structure :-
genai-assistant/
├── app.py              # Main application file
├── requirements.txt    # Dependencies
└── README.md           # Project instructions and description


