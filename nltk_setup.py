# Run this code first on Colab to fix NLTK issues
import nltk
import warnings
warnings.filterwarnings('ignore')

print("Downloading NLTK data for evaluation...")

# Download both old and new tokenizer versions for compatibility
try:
    nltk.download('punkt', quiet=True)
    print("[SUCCESS] Downloaded punkt tokenizer")
except:
    print("[WARNING] Could not download punkt")

try:
    nltk.download('punkt_tab', quiet=True) 
    print("[SUCCESS] Downloaded punkt_tab tokenizer")
except:
    print("[WARNING] Could not download punkt_tab")

try:
    nltk.download('stopwords', quiet=True)
    print("[SUCCESS] Downloaded stopwords")
except:
    print("[WARNING] Could not download stopwords")

print("\nNLTK setup complete! You can now run the evaluation cells.")