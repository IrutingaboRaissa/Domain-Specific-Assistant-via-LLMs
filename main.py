# Pregnancy Healthcare Assistant - Streamlit App
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings
import os
import time
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pregnancy Healthcare Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b9d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #000000;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #000000;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        color: #000000;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">Pregnancy Healthcare Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Domain-Specific AI Assistant for Pregnancy & Maternal Health Questions Only</p>', unsafe_allow_html=True)

# Add domain focus notice
st.info("""**Domain-Specific Assistant**: This AI is specialized for pregnancy and maternal health questions only. 
It will politely redirect questions outside this domain. Ask about pregnancy symptoms, nutrition, exercise, prenatal care, labor, delivery, breastfeeding, and postpartum concerns.""")

# Sidebar for model information
with st.sidebar:
    st.header("Model Information")
    st.write("**Base Model:** TinyLlama-1.1B-Chat-v1.0")
    st.write("**Fine-tuning Method:** LoRA (Parameter-Efficient)")
    st.write("**Training Data:** 2,806 pregnancy Q&A pairs")
    st.write("**Specialized Domain:** Maternal Healthcare")
    
    st.header("Model Stats")
    st.metric("Total Parameters", "1.1B")
    st.metric("Trainable Parameters", "4.5M (0.41%)")
    st.metric("Training Method", "LoRA Fine-tuning")
    
    st.header("Response Settings")
    response_length = st.selectbox(
        "Response Length",
        options=["Short (200 tokens)", "Medium (350 tokens)", "Long (500 tokens)", "Very Long (750 tokens)"],
        index=2,  # Default to "Long"
        help="Longer responses provide more detailed information but take more time to generate"
    )
    
    # Convert selection to token count
    length_mapping = {
        "Short (200 tokens)": 200,
        "Medium (350 tokens)": 350, 
        "Long (500 tokens)": 500,
        "Very Long (750 tokens)": 750
    }
    max_response_tokens = length_mapping[response_length]
    
    st.header("Important Disclaimer")
    st.warning("""
    This AI assistant provides general information only and should NOT replace 
    professional medical advice during pregnancy. Always consult qualified 
    healthcare providers for medical concerns.
    """)

# Configuration - Update these after uploading to Hugging Face  
USE_HUGGINGFACE = False  # Set to True to load from Hugging Face, False for local
HF_MODEL_ID = "Irutingabo/pregnancy-assistant-tinyllama"  # Replace YOUR_USERNAME
LOCAL_MODEL_PATH = "./data/pregnancy-assistant-tinyllama/checkpoint-50"

# Model loading with caching
@st.cache_resource
def load_model():
    """Load the fine-tuned pregnancy assistant model with caching"""
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Choose model path based on configuration
    if USE_HUGGINGFACE:
        MODEL_PATH = HF_MODEL_ID
        st.info(f"Loading model from Hugging Face: {HF_MODEL_ID}")
        # For HF models, no local file check needed
    else:
        MODEL_PATH = LOCAL_MODEL_PATH
        st.info(f"Loading model locally: {LOCAL_MODEL_PATH}")
        # Check if local model exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"Fine-tuned model not found at {MODEL_PATH}")
            st.info("Please ensure you've completed the training process first.")
            return None, None
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model with CPU configuration
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=None,  # Force CPU to avoid memory issues
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Load fine-tuned adapter
        st.info("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(
            base_model, 
            MODEL_PATH,
            device_map=None
        )
        
        st.success(f"Model loaded successfully from {'Hugging Face' if USE_HUGGINGFACE else 'local files'}!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        if USE_HUGGINGFACE:
            st.info("**Troubleshooting tips for Hugging Face loading:**")
            st.info("- The model might need a few minutes after upload to be fully available")
            st.info("- Check if the model repository exists at: https://huggingface.co/Irutingabo/pregnancy-assistant-tinyllama")
            st.info("- Try switching to local loading temporarily by setting USE_HUGGINGFACE = False")
        else:
            st.info("**Troubleshooting tips for local loading:**") 
            st.info("- Ensure the training process completed successfully")
            st.info("- Check that model files exist in the expected directory")
        return None, None

def is_pregnancy_related(question: str) -> bool:
    """Check if the question is related to pregnancy or maternal health"""
    pregnancy_keywords = [
        'pregnan', 'maternity', 'maternal', 'baby', 'fetus', 'fetal', 'birth', 'delivery',
        'labor', 'contractions', 'trimester', 'prenatal', 'postnatal', 'breastfeed', 
        'morning sickness', 'nausea', 'ultrasound', 'obstetrician', 'midwife', 'doula',
        'cervix', 'placenta', 'amniotic', 'conception', 'fertility', 'ovulation',
        'expecting', 'gestational', 'antenatal', 'postpartum', 'lactation', 'nursing',
        'newborn', 'infant', 'c-section', 'vaginal delivery', 'epidural', 'episiotomy',
        'preeclampsia', 'gestational diabetes', 'miscarriage', 'stillbirth', 'ectopic'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in pregnancy_keywords)

def generate_response(model, tokenizer, question: str, max_tokens: int = 500) -> str:
    """Generate response from the fine-tuned model with domain filtering"""
    if model is None or tokenizer is None:
        return "Model not available. Please check the model loading status."
    
    # Check if question is pregnancy-related
    if not is_pregnancy_related(question):
        return """I'm specifically designed to help with pregnancy and maternal health questions. 

Your question doesn't appear to be related to pregnancy or maternal health. I can help you with topics like:
- Pregnancy symptoms and care
- Prenatal nutrition and exercise  
- Labor and delivery questions
- Breastfeeding and postpartum care
- Maternal health concerns

Please ask me a pregnancy-related question, and I'll be happy to help! Remember to always consult with your healthcare provider for personalized medical advice."""
    
    # Format the prompt for pregnancy-related questions
    prompt = f"""### Instruction:
You are a specialized pregnancy healthcare assistant. Answer the following pregnancy-related question with accurate, evidence-based information. Always remind users to consult healthcare providers for personalized medical advice.

{question}

### Response:
"""
    
    try:
        # Tokenize input with increased limits
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Generate response with configurable length
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,  # Use configurable response length
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
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Model loading section
if not st.session_state.model_loaded:
    with st.spinner("Loading your fine-tuned pregnancy assistant..."):
        model, tokenizer = load_model()
        if model is not None:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load model. Please check your setup.")

# Main interface
if st.session_state.model_loaded:
    # Sample questions
    st.header("Sample Pregnancy Questions")
    sample_questions = [
        "Is it safe to eat sushi during pregnancy?",
        "What can help with morning sickness and nausea?",
        "How much weight should I gain during pregnancy?",
        "Can I exercise while pregnant? What's safe?",
        "What medications are safe during pregnancy?",
        "When should I call my doctor during pregnancy?",
        "What foods should I avoid while pregnant?",
        "Is caffeine safe during pregnancy? How much?"
    ]
    
    # Display sample questions as buttons
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        col = cols[i % 2]
        if col.button(f"{question}", key=f"sample_{i}"):
            st.session_state.user_question = question

    st.divider()
    
    # Demo section for domain filtering
    with st.expander("Test Domain-Specific Filtering (Try Non-Pregnancy Questions)"):
        st.write("**Demonstrate that this assistant only responds to pregnancy-related questions:**")
        demo_questions = [
            "How do I know if I have COVID?",
            "What should I know about football?",
            "How do I cook pasta?",
            "What is machine learning?"
        ]
        
        st.write("Try these non-pregnancy questions to see domain filtering in action:")
        demo_cols = st.columns(2)
        for i, demo_q in enumerate(demo_questions):
            col = demo_cols[i % 2]
            if col.button(f"Test: {demo_q}", key=f"demo_{i}"):
                st.session_state.user_question = demo_q

    st.divider()

    # Chat interface
    st.header("Ask Your Pregnancy Question")
    
    # Text input for user question
    user_question = st.text_input(
        "Enter your pregnancy or maternal health question:",
        key="user_input",
        placeholder="e.g., What prenatal vitamins should I take during pregnancy?"
    )
    
    # Use sample question if selected
    if 'user_question' in st.session_state:
        user_question = st.session_state.user_question
        del st.session_state.user_question

    # Generate response button
    col1, col2 = st.columns([1, 5])
    
    with col1:
        ask_button = st.button("Ask Assistant", type="primary")
    
    with col2:
        clear_button = st.button("Clear Chat", type="secondary")

    # Handle clear chat
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

    # Handle question submission
    if ask_button and user_question:
        with st.spinner("Processing your question..."):
            response = generate_response(
                st.session_state.model, 
                st.session_state.tokenizer, 
                user_question,
                max_tokens=max_response_tokens  # Use configurable response length
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": response,
                "timestamp": time.strftime("%H:%M:%S")
            })
        
        st.rerun()

    # Display chat history
    if st.session_state.chat_history:
        st.header("Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You ({chat['timestamp']}):</strong><br>
                {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant message
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Pregnancy Assistant:</strong><br>
                {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()

else:
    # Show loading or error state
    st.error("Please ensure your fine-tuned model is properly trained and saved.")
    st.info("""
    **To use this app:**
    
    1. Complete the training process in the notebook
    2. Ensure the model is saved to: `./data/pregnancy-assistant-tinyllama/checkpoint-50/`
    3. Restart this Streamlit app
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h4>Domain-Specific LLM Fine-Tuning Project</h4>
    <p><strong>Model:</strong> TinyLlama-1.1B with LoRA Fine-tuning</p>
    <p><strong>Domain:</strong> Maternal Healthcare & Pregnancy Support</p>
    <p><strong>Training Method:</strong> Parameter-Efficient Fine-Tuning (PEFT)</p>
    <br>
    <p><em>For educational purposes only. Always consult healthcare professionals for medical advice.</em></p>
</div>
""", unsafe_allow_html=True)