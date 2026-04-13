@echo off
echo =====================================
echo   Air Quality Project Runner
echo =====================================

:: Step 1: Create virtual environment if not exists
if not exist ".aq_venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .aq_venv
)

:: Step 2: Activate virtual environment
call .aq_venv\Scripts\activate

:: Step 3: Install requirements
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

:: Step 4: Run the app
echo Running application...
python app.py

:: Step 5: Pause so window doesn't close
pausee