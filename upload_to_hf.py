"""
Script to upload the fine-tuned pregnancy assistant model to Hugging Face Hub
"""
from huggingface_hub import HfApi, create_repo, upload_folder
import os
import json

def upload_model_to_hf():
    """Upload the fine-tuned model to Hugging Face Hub"""
    
    # Configuration
    MODEL_NAME = "pregnancy-assistant-tinyllama"  # You can change this name
    USERNAME = "Irutingabo"  # Replace with your Hugging Face username
    REPO_ID = f"{USERNAME}/{MODEL_NAME}"
    
    # Path to your model
    MODEL_PATH = "./data/pregnancy-assistant-tinyllama/checkpoint-50"
    
    print("Starting Hugging Face model upload...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Target repository: {REPO_ID}")
    
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path {MODEL_PATH} does not exist!")
        return False
    
    # Required files for LoRA adapter
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors"  
    ]
    
    # Check if required files exist
    for file in required_files:
        if not os.path.exists(os.path.join(MODEL_PATH, file)):
            print(f"Error: Required file {file} not found!")
            return False
    
    try:
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create repository (if it doesn't exist)
        print("Creating repository...")
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False  # Set to True if you want a private repository
        )
        
        # Create a README.md for the model
        readme_content = f"""---
library_name: peft
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
tags:
- generated_from_trainer
- pregnancy
- healthcare
- maternal-health
- domain-specific
model-index:
- name: {MODEL_NAME}
  results: []
---

# Pregnancy Healthcare Assistant

This model is a fine-tuned version of [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) using LoRA (Low-Rank Adaptation) for pregnancy and maternal healthcare questions.

## Model Description

- **Base Model:** TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method:** LoRA (Parameter-Efficient Fine-Tuning)
- **Training Data:** 2,806 pregnancy and maternal health Q&A pairs
- **Domain:** Pregnancy, maternal health, prenatal care, labor, delivery, breastfeeding, postpartum

## Intended Use

This model is designed to provide informational responses to pregnancy and maternal health questions. 
**Important:** This model should NOT replace professional medical advice. Always consult qualified healthcare providers for medical concerns.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "{REPO_ID}")

# Generate response
prompt = "What are the signs of early labor?"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Limitations

- Specialized for pregnancy/maternal health domain only
- General knowledge outside this domain is limited
- Should not be used for medical diagnosis or treatment decisions
- Always recommend consulting healthcare professionals

## Training Details

- Training method: LoRA fine-tuning
- Trainable parameters: 4.5M (0.41% of total model parameters)
- Training dataset: Custom pregnancy Q&A dataset
- Training framework: Transformers + PEFT
"""
        
        # Save README to model directory
        readme_path = os.path.join(MODEL_PATH, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # Upload the model folder
        print("Uploading model files...")
        upload_folder(
            folder_path=MODEL_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload fine-tuned pregnancy assistant model"
        )
        
        print(f"Success! Model uploaded to: https://huggingface.co/{REPO_ID}")
        return True
        
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Verify your Hugging Face username")
        return False

if __name__ == "__main__":
    # Authentication check - skip interactive prompt since we're already authenticated
    print("Starting automatic upload...")
    print("Authentication already completed")
    print("Username already configured: Irutingabo")
    print("\n" + "="*60 + "\n")
    
    # Proceed with upload
    upload_model_to_hf()