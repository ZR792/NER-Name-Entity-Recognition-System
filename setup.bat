@echo off
echo ========================================
echo  NER Project Setup Script
echo ========================================

echo Installing required packages...
pip install pandas==2.2.2 numpy==1.26.4 matplotlib==3.9.0 openpyxl==3.1.2 scikit-learn==1.4.2 spacy==3.7.4 sklearn-crfsuite==0.3.6 streamlit==1.35.0 seaborn==0.13.2 joblib==1.4.2

echo Installing SpaCy model...
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

echo Installing Jupyter...
pip install jupyter notebook ipykernel

echo Registering Jupyter kernel...
python -m ipykernel install --user --name=ner_env --display-name "Python 3.11 NER Project"

echo ========================================
echo  Setup Complete!
echo  Now open VS Code and select kernel:
echo  "Python 3.11 NER Project"
echo ========================================
pause
