import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time
import warnings
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
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .example-question {
        background-color: #f8f9fa;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        cursor: pointer;
        border: 1px solid #dee2e6;
    }
    .example-question:hover {
        background-color: #e9ecef;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #8bc34a;
    }
</style>
""", unsafe_allow_html=True)

# Model loading function with caching
@st.cache_resource
def load_model():
    """Load the fine-tuned pregnancy assistant model."""
    try:
        BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        FINETUNED_MODEL_PATH = "pregnancy-assistant-tinyllama"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load fine-tuned adapter
        model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Generate response function
def generate_response(model, tokenizer, question, max_length=300, temperature=0.7):
    """Generate response from the pregnancy assistant."""
    if model is None or tokenizer is None:
        return "Model not loaded. Please check the model path."
    
    # Format prompt
    prompt = f"""### Instruction:
You are a pregnancy healthcare assistant. Answer the following question with accurate, evidence-based information. Always remind users to consult healthcare providers for medical concerns.

{question}

### Response:
"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode and extract response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in full_response:
        response = full_response.split("### Response:")[-1].strip()
    else:
        response = full_response.strip()
    
    return response

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">Pregnancy Healthcare Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI companion for pregnancy questions and maternal health information</p>', unsafe_allow_html=True)
    
    # Disclaimer box
    st.markdown("""
    <div class="disclaimer-box">
        <h4 style="margin-top: 0;">IMPORTANT MEDICAL DISCLAIMER</h4>
        <p style="margin-bottom: 0;">
            This assistant provides <strong>general information only</strong> and is NOT intended to replace 
            professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
            providers regarding any medical conditions or concerns during pregnancy. 
            <br><br>
            <strong>In case of medical emergency, call emergency services immediately (911 in the US).</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("How to Use")
        
        st.markdown("""
        ### Getting Started
        
        1. **Type your question** in the text area below
        2. **Click "Get Answer"** to receive a response
        3. **Review the information** provided
        4. **Ask follow-up questions** as needed
        
        ### Tips for Best Results
        
        - Be specific in your questions
        - Ask one question at a time
        - Provide relevant context if needed
        - Always verify with your healthcare provider
        
        ### What I Can Help With
        
        - Nutrition and diet during pregnancy
        - Exercise and physical activity
        - Common symptoms and discomforts
        - Medications and supplements
        - Prenatal care and monitoring
        - Labor and delivery preparation
        - Postpartum care
        - General pregnancy questions
        
        ### What to Avoid Asking
        
        - Medical diagnosis of conditions
        - Emergency medical situations
        - Prescription medication advice
        - Complex medical procedures
        - Legal or financial advice
        """)
        
        st.divider()
        
        st.markdown("### Configuration")
        
        # Temperature slider
        temperature = st.slider(
            "Response Creativity",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make responses more creative but less focused"
        )
        
        max_length = st.slider(
            "Response Length",
            min_value=100,
            max_value=500,
            value=300,
            step=50,
            help="Maximum number of tokens to generate"
        )
        
        st.divider()
        
        st.markdown("""
        ### About This Project
        
        This pregnancy healthcare assistant was created by fine-tuning **TinyLlama-1.1B** 
        on pregnancy-specific medical Q&A data using **LoRA (Low-Rank Adaptation)**.
        
        **Key Features:**
        - Specialized in pregnancy and maternal health
        - Trained on evidence-based medical information
        - Fast response time (2-3 seconds)
        - Efficient inference using 4-bit quantization
        
        **Technology Stack:**
        - Model: TinyLlama-1.1B-Chat
        - Fine-tuning: LoRA/PEFT
        - Framework: Transformers, PyTorch
        - Interface: Streamlit
        
        **Resources:**
        - [GitHub Repository](https://github.com/YOUR_USERNAME/Domain-Specific-Assistant-via-LLMs)
        - [Demo Video](YOUTUBE_LINK)
        
        ---
        
        *Created as part of Domain-Specific LLM Fine-Tuning Project*  
        *February 2026*
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask Your Question")
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Question input
        question = st.text_area(
            "Enter your pregnancy-related question:",
            height=100,
            placeholder="e.g., Is it safe to exercise during pregnancy?",
            key="question_input"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            ask_button = st.button("Get Answer", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_button = st.button("Clear Chat", use_container_width=True)
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        # Process question
        if ask_button and question.strip():
            with st.spinner("Thinking..."):
                # Load model
                model, tokenizer = load_model()
                
                if model is not None:
                    # Generate response
                    start_time = time.time()
                    response = generate_response(
                        model, 
                        tokenizer, 
                        question, 
                        max_length=max_length,
                        temperature=temperature
                    )
                    end_time = time.time()
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'response': response,
                        'time': end_time - start_time
                    })
                else:
                    st.error("Failed to load model. Please check the model path and try again.")
        
        # Display chat history
        if st.session_state.chat_history:
            st.divider()
            st.subheader("Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>Question:</strong> {chat['question']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Answer:</strong> {chat['response']}
                        <br><br>
                        <small><em>Response time: {chat['time']:.2f} seconds</em></small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
    
    with col2:
        st.subheader("Example Questions")
        
        st.markdown("""
        <div class="info-box">
            <p><strong>Click on any question below to try it:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        example_questions = [
            "Is it safe to eat sushi during pregnancy?",
            "What can help with morning sickness?",
            "How much weight should I gain during pregnancy?",
            "Can I drink coffee while pregnant?",
            "What prenatal vitamins should I take?",
            "Is it safe to exercise during pregnancy?",
            "What foods should I avoid during pregnancy?",
            "How can I relieve back pain during pregnancy?",
            "When should I call my doctor during pregnancy?",
            "What are signs of preterm labor?",
            "Can I travel during pregnancy?",
            "Is it safe to take acetaminophen while pregnant?",
            "How much water should I drink daily?",
            "What are normal first trimester symptoms?",
            "Can I sleep on my back during pregnancy?",
        ]
        
        for i, example in enumerate(example_questions):
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.question_input = example
                st.rerun()
        
        st.divider()
        
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0;">Emergency Contacts</h4>
            <p>
            <strong>Emergency:</strong> 911<br>
            <strong>Poison Control:</strong> 1-800-222-1222<br>
            <strong>National Maternal Hotline:</strong> 1-833-TLC-MAMA
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0;">Helpful Resources</h4>
            <ul style="margin-bottom: 0;">
                <li><a href="https://www.acog.org/" target="_blank">ACOG - Pregnancy Info</a></li>
                <li><a href="https://www.cdc.gov/pregnancy/" target="_blank">CDC - Pregnancy</a></li>
                <li><a href="https://www.marchofdimes.org/" target="_blank">March of Dimes</a></li>
                <li><a href="https://www.nichd.nih.gov/health/topics/pregnancy" target="_blank">NICHD - Pregnancy Health</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>
            <strong>Remember:</strong> This AI assistant is a helpful resource, but it cannot replace 
            the expertise and personalized care of your healthcare provider. Always discuss any concerns 
            or medical conditions with qualified professionals.
        </p>
        <p style="margin-top: 1rem; font-size: 0.9rem;">
            Domain-Specific LLM Fine-Tuning Project | February 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
