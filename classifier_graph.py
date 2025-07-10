from langgraph.graph import StateGraph, END
from typing import TypedDict
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# === Define the input/output state for LangGraph ===
class ClassificationState(TypedDict):
    input_text: str
    prediction: str
    confidence: float
    used_fallback: bool

# === Label mapping for readable output ===
LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE"
}

# === Load primary model (DistilBERT fine-tuned) ===
print("📦 Loading fine-tuned DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSequenceClassification.from_pretrained("./model")
pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# === Node: Inference ===
def run_inference(state: ClassificationState) -> ClassificationState:
    result = pipe(state["input_text"], return_all_scores=True)[0]
    best = max(result, key=lambda x: x['score'])
    readable_label = LABEL_MAP.get(best["label"], best["label"])  # fallback if unknown
    return {
        **state,
        "prediction": readable_label,
        "confidence": best["score"],
        "used_fallback": False
    }

# === Condition function ===
CONFIDENCE_THRESHOLD = 0.6

def confidence_condition(state: ClassificationState) -> str:
    return "high_confidence" if state["confidence"] >= CONFIDENCE_THRESHOLD else "low_confidence"

# === Node: Fallback logic (User Clarification) ===
def clarify_with_user(state: ClassificationState) -> ClassificationState:
    print("\n Confidence is low. Let's clarify before deciding.")
    user_input = input("Did you mean this to be a POSITIVE or NEGATIVE statement?\n> ").strip().upper()

    while user_input not in ["POSITIVE", "NEGATIVE"]:
        user_input = input("Please type either 'POSITIVE' or 'NEGATIVE':\n> ").strip().upper()

    return {
        **state,
        "prediction": user_input,
        "confidence": 1.0,  # Treat manual clarification as high confidence
        "used_fallback": True
    }

# === Define and build the DAG ===
graph = StateGraph(ClassificationState)

graph.add_node("Inference", run_inference)
graph.add_node("Fallback", clarify_with_user)

graph.set_entry_point("Inference")
graph.add_conditional_edges(
    source="Inference",
    path=confidence_condition,
    path_map={
        "high_confidence": END,
        "low_confidence": "Fallback"
    }
)
graph.add_edge("Fallback", END)

classifier_app = graph.compile()
print("✅ LangGraph DAG compiled and ready!")
