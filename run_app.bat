@echo off
echo 🤰 Starting Pregnancy Healthcare Assistant...
echo ===================================================

echo 📦 Installing/updating requirements...
pip install -r streamlit_requirements.txt --quiet

echo 🚀 Launching Streamlit app...
echo.
echo 💻 The app will open in your default browser
echo 🔗 Local URL: http://localhost:8501
echo.
echo ⏹️  Press Ctrl+C in this window to stop the app
echo.

streamlit run pregnancy_assistant_app.py