# cli_interface.py
import sys
import csv
import os
from datetime import datetime
from classifier_graph import classifier_app, ClassificationState

LOG_FILE = "classifier_log.csv"

# Initialize log file with headers if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Timestamp", "Input Text", "Prediction", "Confidence", "Used Fallback"])

def log_classification(state: ClassificationState):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            timestamp,
            state["input_text"],
            state["prediction"],
            f"{state['confidence']:.2f}",
            state["used_fallback"]
        ])

def classify_text(text: str):
    print("\n🔍 Classifying input text...")
    result = classifier_app.invoke({"input_text": text})
    print("\n=== Classification Result ===")
    print(f"Label:         {result['prediction']}")
    print(f"Confidence:    {result['confidence']:.2f}")
    print(f"Used Fallback: {result['used_fallback']}")
    print("============================\n")
    log_classification(result)

if __name__ == "__main__":
    print("\n🤖 Welcome to the LangGraph Classifier CLI!")
    print("Type a sentence to classify (or type 'exit' to quit):")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            sys.exit()
        elif user_input:
            classify_text(user_input)
