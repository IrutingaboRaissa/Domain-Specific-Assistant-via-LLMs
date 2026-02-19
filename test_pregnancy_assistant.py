# Test Your Fine-Tuned Pregnancy Healthcare Assistant
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./data/pregnancy-assistant-tinyllama/checkpoint-50"

print("Loading your fine-tuned Pregnancy Healthcare Assistant...")
print("="*60)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# Load your fine-tuned LoRA adapter
print("Loading your fine-tuned adapter...")
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

print("Model loaded successfully!")
print("="*60)

def ask_pregnancy_question(question: str, max_length: int = 200) -> str:
    """
    Ask your pregnancy assistant a question
    """
    prompt = f"""### Instruction:
You are a pregnancy healthcare assistant. Answer the following question with accurate, evidence-based information. Always remind users to consult healthcare providers for medical concerns.

{question}

### Response:
"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("### Response:")[-1].strip()
    
    return response

# Test questions for your pregnancy assistant
test_questions = [
    "Is it safe to eat sushi during pregnancy?",
    "What can I do to relieve morning sickness?", 
    "How much weight should I gain during pregnancy?",
    "Can I exercise while pregnant?",
    "What medications are safe to take during pregnancy?",
    "When should I call my doctor during pregnancy?",
    "What foods should I avoid while pregnant?",
    "Is coffee safe during pregnancy?",
]

print("TESTING YOUR PREGNANCY ASSISTANT")
print("="*60)

for i, question in enumerate(test_questions, 1):
    print(f"\nQuestion {i}: {question}")
    print("-" * 50)
    
    try:
        answer = ask_pregnancy_question(question)
        print(f"Assistant: {answer}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("-" * 50)

print("\nTesting complete!")
print("\nTry asking your own questions:")
print("   Use: ask_pregnancy_question('Your question here')")
print("\nRemember: This is for educational purposes only.")
print("   Always consult healthcare professionals for medical advice!")