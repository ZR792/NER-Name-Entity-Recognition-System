# NER System — Sports & Political News
### Introduction to Data Science | SMIU

---

## SETUP (Run Once)

1. Make sure Python 3.11 is installed
2. Double-click `setup.bat` OR run in terminal:
   ```
   pip install -r requirements.txt
   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
   ```

---

## PROJECT STRUCTURE

```
ner_project/
├── data/
│   └── NER_Dataset_Complete_Tagged.xlsx   ← your labeled data
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
├── models/                                ← generated after training
├── app.py                                 ← Streamlit UI
├── requirements.txt
└── README.md
```

---

## HOW TO RUN

### Step 1 — Run notebooks in order (in VS Code)
Open each notebook and run all cells:
1. `01_data_cleaning.ipynb`     → generates cleaned_data.csv
2. `02_eda.ipynb`               → generates EDA charts
3. `03_feature_engineering.ipynb` → generates iob_data.json
4. `04_model_training.ipynb`    → trains both models
5. `05_evaluation.ipynb`        → generates comparison charts

### Step 2 — Run Streamlit UI
```
streamlit run app.py
```

---

## MODELS
- Model 1: SpaCy NER (neural deep learning)
- Model 2: CRF — Conditional Random Field (classical ML)
- Entity Types: PERSON and LOCATION
