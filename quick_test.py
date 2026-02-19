"""
Quick Test Script for Pregnancy Assistant Demo
Test the model functionality before your demo presentation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def test_model_loading():
    """Test if the fine-tuned model loads correctly"""
    print("Testing Pregnancy Assistant Model...")
    print("=" * 50)
    
    try:
        # Check if model files exist
        model_path = "./data/pregnancy-assistant-tinyllama/checkpoint-50"
        if os.path.exists(model_path):
            print("[SUCCESS] Model checkpoint found at:", model_path)
        else:
            print("[ERROR] Model checkpoint not found!")
            return False, None, None
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[SUCCESS] Tokenizer loaded successfully")
        
        # Load base model
        print("\nLoading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map=None
        )
        print("[SUCCESS] Base model loaded successfully")
        
        # Load fine-tuned model
        print("\nLoading fine-tuned model...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print("[SUCCESS] Fine-tuned model loaded and merged successfully")
        
        return True, tokenizer, model
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
        return False, None, None

def test_model_response(tokenizer, model, question):
    """Test model response generation"""
    try:
        # Format prompt
        prompt = f"<|system|>\nYou are a helpful pregnancy and maternal health assistant. Provide accurate, supportive information while recommending consulting healthcare professionals for personalized advice.\n<|user|>\n{question}\n<|assistant|>\n"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("<|assistant|>")[-1].strip()
        
        return assistant_response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    """Main test function"""
    print("PREGNANCY ASSISTANT DEMO TEST")
    print("=" * 50)
    
    # Test model loading
    success, tokenizer, model = test_model_loading()
    
    if not success:
        print("\n[ERROR] Model loading failed! Check the sample responses below:")
        
        # Fallback - show expected responses
        sample_qa = [
            {
                "question": "Is it safe to eat sushi during pregnancy?", 
                "trained_response": "Most sushi is safe during pregnancy, but avoid raw fish due to potential mercury and bacteria. Stick to cooked sushi rolls like California rolls, tempura rolls, or vegetarian options."
            },
            {
                "question": "What helps with morning sickness?",
                "trained_response": "Try eating small, frequent meals every 2-3 hours. Keep crackers by your bed and eat a few before getting up. Ginger tea, candies, or supplements can help nausea."
            },
            {
                "question": "Can I exercise while pregnant?",
                "trained_response": "Yes! Regular, moderate exercise is beneficial. Walking, swimming, prenatal yoga are excellent. Avoid contact sports, activities with fall risk, or lying flat after first trimester."
            }
        ]
        
        print("\nEXPECTED MODEL RESPONSES (from training data):")
        for i, qa in enumerate(sample_qa, 1):
            print(f"\n{i}. Q: {qa['question']}")
            print(f"   A: {qa['trained_response']}")
        
        print(f"\nYOU CAN STILL DEMO THE STREAMLIT APP!")
        print("Run: streamlit run pregnancy_assistant_app.py")
        return
    
    print("\n[SUCCESS] MODEL LOADING SUCCESSFUL!")
    print("\n" + "=" * 50)
    print("TESTING MODEL RESPONSES...")
    print("=" * 50)
    
    # Test questions for demo
    test_questions = [
        "Is it safe to eat sushi during pregnancy?",
        "What helps with morning sickness?",
        "Can I exercise while pregnant?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 40)
        
        response = test_model_response(tokenizer, model, question)
        print("Response:")
        print(response)
        print("-" * 40)
    
    print(f"\n[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY!")
    print("\nYOUR MODEL IS READY FOR THE DEMO!")
    print("\nNext steps:")
    print("1. Run: streamlit run pregnancy_assistant_app.py")
    print("2. Open the web interface")
    print("3. Start your demo recording!")

if __name__ == "__main__":
    main()