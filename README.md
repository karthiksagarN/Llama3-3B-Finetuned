# Llama 3 3B Finetuned

## Financial Transaction Categorizer

This repository contains a **Llama 3 3B Instruct** model fine-tuned using **QLoRA** to classify bank transaction descriptions into 12 distinct financial categories. The model is designed to be a powerful, lightweight solution for personal finance management applications, automated bookkeeping, and spending analysis.

This model was trained on a custom synthetic dataset of transaction descriptions. It takes a raw transaction string as input and outputs the most likely spending category.

## Model Description

- **Base Model:** `meta-llama/Meta-Llama-3-8B-Instruct` (The script uses a 3B variant, but this is a common public equivalent. You can update this to the exact base model ID if available on the Hub).
- **Fine-tuning Method:** QLoRA (Quantization with Low-Rank Adapters) for memory-efficient training.
- **Task:** Text Classification (formatted as a Causal Language Modeling task).
- **Categories (12):** `Education`, `Travel & Transport`, `Groceries`, `Miscellaneous`, `Bills & Utilities`, `Health & Fitness`, `Shopping`, `Entertainment`, `Investments`, `Income`, `Food & Drinks`, `Withdrawals`.

## How to Use

To use this model, you need to load the base Llama 3 model and then apply the fine-tuned LoRA adapters from this repository. Make sure you have the `transformers`, `peft`, `accelerate`, and `bitsandbytes` libraries installed.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Define the base model and the adapter path (your Hugging Face repo)
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
adapter_id = "karthiksagarn/llama3-3.2b-finetuned-financial" # Replace with your HF repo ID

# Load the base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    # token="YOUR_HUGGINGFACE_TOKEN" # Add your token if needed
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_id)

# Load the LoRA adapter and merge it onto the base model
model = PeftModel.from_pretrained(base_model, adapter_id)

# Create a text-generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# --- Inference ---
def classify_transaction(description):
    # The list of labels the model was trained on
    labels = [
        "Education", "Travel & Transport", "Groceries", "Miscellaneous", 
        "Bills & Utilities", "Health & Fitness", "Shopping", "Entertainment", 
        "Investments", "Income", "Food & Drinks", "Withdrawals"
    ]
    
    prompt = (
        f"Classify the following bank transaction into one of these categories:\n"
        f"{', '.join(labels)}\n\n"
        f"Description: {description}\n\nCategory:"
    )
    
    # Generate the output
    output = pipe(prompt, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    # Clean and parse the output
    generated_text = output[0]["generated_text"]
    category = generated_text.split("Category:")[-1].strip().split("\n")[0].strip()
    
    # Fallback to ensure a valid category is returned
    if category in labels:
        return category
    return "Miscellaneous" # Fallback if parsing fails

# --- Example Usage ---
transaction1 = "Sent Rs.510.00 From ABCD Bank A/C **** To Zomato Limited On 10/01/29"
transaction2 = "UPI Payment to Amazon for new headphones"
transaction3 = "Salary credited from Awesome Tech Inc."
transaction4 = "Recharge of Airtel mobile number"

print(f"'{transaction1}' -> Category: {classify_transaction(transaction1)}")
print(f"'{transaction2}' -> Category: {classify_transaction(transaction2)}")
print(f"'{transaction3}' -> Category: {classify_transaction(transaction3)}")
print(f"'{transaction4}' -> Category: {classify_transaction(transaction4)}")

# Expected Output:
# 'Sent Rs.510.00...' -> Category: Food & Drinks
# 'UPI Payment to Amazon...' -> Category: Shopping
# 'Salary credited from...' -> Category: Income
# 'Recharge of Airtel...' -> Category: Bills & Utilities
```

## Training & Evaluation

### Training Procedure

The model was fine-tuned on a private, balanced dataset of synthetic bank transactions.

- **Quantization:** 4-bit (`nf4`) with double quantization.
- **LoRA Configuration:**
  - Rank (`r`): **16**
  - Alpha (`lora_alpha`): **32**
  - Target Modules: `q_proj`, `v_proj`
  - Dropout: `0.05`
- **Training Hyperparameters:**
  - Epochs: **3**
  - Learning Rate: `2e-4`
  - Optimizer: `adamw_torch`
  - Effective Batch Size: **16** (2 per device * 8 accumulation steps)
  - Scheduler: Linear warmup

### Training and Validation Loss
The model showed consistent improvement over 3 epochs, with the validation loss decreasing steadily, indicating good generalization.

| Epoch | Training Loss | Validation Loss |
|:-----:|:-------------:|:---------------:|
| 1     | 0.464400      | 0.448071        |
| 2     | 0.396200      | 0.402424        |
| 3     | 0.394500      | 0.399548        |


### Evaluation Results

The model achieves an overall accuracy of **86.03%** on the held-out test set.

**Classification Report:**

| Category             | Precision | Recall | F1-Score | Support |
|:---------------------|:---------:|:------:|:--------:|:-------:|
| **Education** | 1.00      | 0.35   | 0.52     | 125     |
| **Travel & Transport** | 1.00      | 1.00   | 1.00     | 125     |
| **Groceries** | 0.95      | 1.00   | 0.98     | 125     |
| **Miscellaneous** | 0.42      | 0.90   | 0.58     | 125     |
| **Bills & Utilities**| 1.00      | 1.00   | 1.00     | 125     |
| **Health & Fitness** | 1.00      | 1.00   | 1.00     | 125     |
| **Shopping** | 0.93      | 0.62   | 0.74     | 125     |
| **Entertainment** | 0.83      | 0.77   | 0.80     | 125     |
| **Investments** | 0.85      | 0.75   | 0.80     | 126     |
| **Income** | 1.00      | 0.97   | 0.98     | 126     |
| **Food & Drinks** | 0.99      | 0.97   | 0.98     | 126     |
| **Withdrawals** | 0.95      | 1.00   | 0.97     | 125     |
|                      |           |        |          |         |
| **Macro Avg** | 0.91      | 0.86   | 0.86     | 1503    |
| **Weighted Avg** | 0.91      | 0.86   | 0.86     | 1503    |

**Confusion Matrix:**
```
[[ 44   0   6  54   0   0   3  16   1   0   1   0]  -> Education
 [  0 125   0   0   0   0   0   0   0   0   0   0]  -> Travel & Transport
 [  0   0 125   0   0   0   0   0   0   0   0   0]  -> Groceries
 [  0   0   0 113   0   0   0   2   3   0   0   7]  -> Miscellaneous
 [  0   0   0   0 125   0   0   0   0   0   0   0]  -> Bills & Utilities
 [  0   0   0   0   0 125   0   0   0   0   0   0]  -> Health & Fitness
 [  0   0   0  42   0   0  77   2   4   0   0   0]  -> Shopping
 [  0   0   0  18   0   0   3  96   8   0   0   0]  -> Entertainment
 [  0   0   0  32   0   0   0   0  94   0   0   0]  -> Investments
 [  0   0   0   4   0   0   0   0   0 122   0   0]  -> Income
 [  0   0   0   4   0   0   0   0   0   0 122   0]  -> Food & Drinks
 [  0   0   0   0   0   0   0   0   0   0   0 125]] -> Withdrawals
```

### Performance Analysis & Limitations

- **High-Performing Categories:** The model is extremely reliable for clear, unambiguous categories like `Travel & Transport`, `Groceries`, `Bills & Utilities`, and `Health & Fitness`, achieving perfect or near-perfect precision and recall.

- **Areas for Improvement:**
  - **`Education` vs. `Miscellaneous`:** The model's biggest weakness is distinguishing `Education` transactions. The recall is very low (0.35), and the confusion matrix shows that **54 out of 125** education-related transactions were incorrectly classified as `Miscellaneous`. This suggests the descriptions for these categories may be too similar or that the model needs more distinct examples.
  - **`Miscellaneous` Precision:** The precision for `Miscellaneous` is low (0.42) because many other classes are incorrectly "dumped" into it when the model is uncertain.
  - **Ambiguous Commercial Categories:** There is some confusion between `Shopping`, `Entertainment`, and `Investments`, which is expected as transaction descriptions in these areas can be vague (e.g., a payment to a large conglomerate like Amazon could be for goods, services, or media).

This model provides a strong baseline for transaction classification but could be improved with a more diverse dataset, especially with clearer examples to differentiate the weaker categories.