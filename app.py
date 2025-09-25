from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import difflib

token = os.getenv("HF_TOKEN")

# -------------------------
# 1. Config
# -------------------------
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"   # or the one you trained on
LORA_MODEL = "./saved_model"  # your LoRA adapter directory

labels = [
    "Food & Drinks",
    "Travel & Transport",
    "Shopping",
    "Bills & Utilities",
    "Entertainment",
    "Health & Fitness",
    "Groceries",
    "Education",
    "Income",
    "Investments",
    "Withdrawals",
    "Miscellaneous"
]

# Few-shot examples
FEW_SHOT_EXAMPLES = [
    ("Sent Rs.510.00 From HDFC Bank A/C *0552 To Swiggy Limited", "Food & Drinks"),
    ("Salary credited from ACME Technologies", "Income"),
    ("Paid electricity bill online", "Bills & Utilities"),
    ("Bought vegetables at the local market", "Groceries"),
    ("ATM cash withdrawal 3000", "Withdrawals"),
]
examples_text = "\n\n".join(f"Description: {d}\nCategory: {c}" for d, c in FEW_SHOT_EXAMPLES)

# -------------------------
# 2. Load model + tokenizer
# -------------------------
print("Loading base model from HF Hub...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    use_auth_token=token,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_MODEL)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# -------------------------
# 3. FastAPI App
# -------------------------
app = FastAPI()

class Transaction(BaseModel):
    description: str

@app.get("/")
def home():
    return {"message": "Bank Transaction Classifier is running 🚀"}

@app.post("/predict")
def predict(tx: Transaction):
    prompt = (
        "You are a classifier. Return EXACTLY one of the labels below and NOTHING ELSE.\n\n"
        f"Labels: {', '.join(labels)}\n\n"
        "Examples:\n"
        f"{examples_text}\n\n"
        "Now classify the following:\n"
        f"Description: {tx.description}\n"
        "Category:"
    )

    out = pipe(prompt, max_new_tokens=8, do_sample=False, temperature=0.0)
    generated = out[0]["generated_text"].split("Category:")[-1].strip()
    generated = generated.split("\n")[0].strip()

    # Post-processing
    if generated in labels:
        return {"predicted_category": generated}

    for lbl in labels:
        if lbl.lower() in generated.lower() or generated.lower() in lbl.lower():
            return {"predicted_category": lbl}

    match = difflib.get_close_matches(generated, labels, n=1, cutoff=0.55)
    if match:
        return {"predicted_category": match[0]}

    return {"predicted_category": "Miscellaneous"}
