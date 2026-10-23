# Bank Customer Churn Prediction

A Streamlit application that trains a Random Forest model to estimate bank customer churn risk. It displays model evaluation metrics, churn analysis, feature importance, individual predictions, and scenario-based risk estimates.

## Setup Instructions

1. Clone or download this project folder.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

For Command Prompt, activate it with:

```bat
venv\Scripts\activate.bat
```

4. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

5. Run the app:

```bash
python -m streamlit run Bankchurn.py
```

Streamlit will print a local URL to open in your browser.

## Project Files

- `Bankchurn.py`: Streamlit app and model workflow.
- `Churn_Modelling.csv`: Dataset used by the app.
- `European_Bank.csv`: Additional bank dataset included in the project.
- `requirements.txt`: Python package dependencies.

## Notes

- Python 3.10 or newer is recommended.
- Keep `Churn_Modelling.csv` in the project directory. If it is missing or empty, the app attempts to download a replacement from the configured GitHub dataset URL.
- The model is trained when the Streamlit app runs; no separately trained model file is required.