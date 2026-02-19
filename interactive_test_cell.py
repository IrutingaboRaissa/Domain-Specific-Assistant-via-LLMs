# INTERACTIVE TESTING CELL - Add this to your notebook!

# Test Your Fine-Tuned Pregnancy Assistant
def test_pregnancy_assistant():
    print("PREGNANCY HEALTHCARE ASSISTANT - INTERACTIVE TEST")
    print("="*60)
    
    # Load your trained model (adapt paths as needed)
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    OUTPUT_DIR = "./pregnancy-assistant-tinyllama/checkpoint-50"  # or your model path
    
    if 'model' not in globals():
        print("Loading your fine-tuned model...")
        
        # Load tokenizer
        global tokenizer, model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model  
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        # Load fine-tuned adapter
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        print("Model loaded successfully!")
    else:
        print("Using previously loaded model")
    
    def ask_question(question: str) -> str:
        prompt = f"""### Instruction:
You are a pregnancy healthcare assistant. Answer the following question with accurate, evidence-based information. Always remind users to consult healthcare providers for medical concerns.

{question}

### Response:
"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("### Response:")[-1].strip()
    
    # Quick tests
    test_questions = [
        "Is it safe to drink coffee during pregnancy?",
        "What are the signs of preterm labor?", 
        "Can I eat seafood while pregnant?",
    ]
    
    print(f"\nQUICK TESTS:")
    for i, q in enumerate(test_questions, 1):
        print(f"\n{i}. Q: {q}")
        answer = ask_question(q)
        print(f"   A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
    
    print(f"\n{'='*60}")
    print("CUSTOM TESTING - Try your own questions:")
    print("   answer = ask_question('Your question here')")
    print("   print(answer)")
    
    # Return the function for interactive use
    return ask_question

# Run the test
ask_question = test_pregnancy_assistant()