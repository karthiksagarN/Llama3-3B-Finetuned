from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -------------------------
# 1. Config
# -------------------------
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"   # or the one you trained on
LORA_MODEL = "./saved_model"  # your LoRA adapter repo on HF

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
# -------------------------
# 2. Load model + tokenizer
# -------------------------
print("Loading base model from HF Hub...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
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
def predict(text: Transaction):
        prompt = (
        f"Classify the following bank transaction into one of these categories:\n"
        f"{', '.join(labels)}\n\n"
        f"Description: {text}\n\nCategory:"
        )
        output = pipe(prompt, max_new_tokens=20, do_sample=False)
        generated = output[0]["generated_text"].split("Category:")[-1].strip()
        generated = generated.split("\n")[0].strip()
        return generated
