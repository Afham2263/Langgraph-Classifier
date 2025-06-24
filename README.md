LangGraph Sentiment Classifier

A robust and self-healing text classification system built with LangGraph, fine-tuned transformers, and fallback logic to ensure accurate and reliable sentiment prediction.

🎯 Objective

This project classifies text (e.g., movie reviews) as either POSITIVE or NEGATIVE using a fine-tuned DistilBERT model. When prediction confidence is low, it triggers a fallback strategy to recover gracefully using either user clarification or a backup model.

🛠️ Features

✅ Fine-tuned transformer model for sentiment classification

🔁 LangGraph DAG with conditional fallback logic

🔍 Confidence-based prediction acceptance or clarification

🧠 Optional zero-shot fallback model

🧾 CLI interface for human-in-the-loop review

📊 Logging-ready design for tracking model decisions (expandable)

📁 Project Structure

langgraph_classifier/
├── classifier_graph.py       # LangGraph DAG and node logic
├── cli_interface.py          # User CLI for interaction
├── train_model.py            # Fine-tuning script
├── load_data.py              # Dataset loading & preprocessing
├── requirements.txt          # Dependencies
├── .gitignore                # Ignores large and unnecessary files
└── README.md                 # You are here

📦 Installation

git clone https://github.com/<your-username>/Langgraph-Classifier.git
cd Langgraph-Classifier
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate
pip install -r requirements.txt

🚀 Usage

🔧 Model Setup

The trained model isn't pushed to GitHub due to size. Download it from:

🔗 Google Drive Model Folder

Unzip it and place the folder as ./model in the project directory.

🧪 Run CLI Classifier

python cli_interface.py

You'll be greeted with:

🤖 Welcome to the LangGraph Classifier CLI!
> This movie was awful

If confidence is low, you'll be prompted for clarification:

🤔 Confidence is low. Let's clarify before deciding.
Did you mean this to be a POSITIVE or NEGATIVE statement?

🧠 How It Works

🔄 DAG Flow

[Input Text] → InferenceNode
       ↳ High Confidence → ✅ END
       ↳ Low Confidence → FallbackNode (user clarification)
                                  ↳ END

InferenceNode: Runs classification using fine-tuned DistilBERT.

ConfidenceCheck: Inline conditional logic (threshold = 0.6).

FallbackNode: Prompts user to confirm sentiment manually.

🏋️ Model Training

Want to train your own? Run:

python train_model.py

This script fine-tunes DistilBERT using the IMDb dataset via HuggingFace Datasets, with LoRA for parameter-efficient training.

📜 Requirements

Install dependencies from requirements.txt. Key packages include:

transformers
peft
langgraph
accelerate
torch
scikit-learn

📹 Demo

🧪 CLI + fallback logic showcased in:
demo.mp4 (or add a YouTube link)

📚 References

LangGraph Docs

Hugging Face Transformers

LoRA Paper

👨‍💻 Author
by Afham

📜 License

MIT License. 
