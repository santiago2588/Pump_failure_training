![Workshop Banner](https://res.cloudinary.com/dtradpei6/image/upload/data_bfnxm8.jpg)
[![GitHub Pages](https://img.shields.io/badge/View%20Site-GitHub%20Pages-blue?logo=github)](https://santiago2588.github.io/Pump_failure_training/)

# Predictive Maintenance: Failure Classifier - Workshop 🧪

## Overview 📋
This hands-on workshop guides participants through building and evaluating various classification models to predict equipment failures. Participants will learn by working through a series of Jupyter notebooks, from data loading and preparation to model training and evaluation, and finally deploying a model with a Streamlit application.

## What You’ll Learn 🧠
* Loading, inspecting, and cleaning real-world data.
* Performing data transformations to prepare data for modeling.
* Establishing a baseline model for comparison.
* Implementing and fine-tuning tree-based classification models:
  * Random Forest
  * XGBoost
  * LightGBM (LGBM)
* Using AutoML with PyCaret to find the best model.
* Evaluating model performance using appropriate metrics.
* Understanding the workflow of a machine learning project from data to model.
* Deploying a model with a Streamlit application.

## Getting Started 🛠️
✅ **Recommended Platform:** Google Colab. Google Colab provides a free, interactive environment that's ideal for this workshop. No local installation is required!

### What You Need:
* A Google account.
* A reliable internet connection.

### Running the Notebooks in Colab:
1. **Access the Notebooks:**
   - Open the main GitHub repository page for this workshop.
   - Navigate to the `soluciones/` directory.
2. **Open in Colab:**
   - Click on a notebook file (e.g., `01_load_and_clean_data.ipynb`).
   - Look for an "Open in Colab" badge/button at the top of the notebook preview on GitHub. Click it.
   - Alternatively, if the badge isn't available:
     - On the GitHub notebook page, click the "Raw" button. Copy the URL from your browser's address bar.
     - Open Google Colab (https://colab.research.google.com/).
     - Select `File > Open notebook`.
     - Choose the "GitHub" tab, paste the URL, and press Enter.
3. **Install Dependencies (in Colab):**
   - Once a notebook is open in Colab, the first code cell in many notebooks will be for installing necessary libraries from the `requirements.txt` file.
   - Run this cell by pressing `Shift+Enter` or clicking the play button.

### Running the Streamlit Application Locally:
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```bash
   streamlit run main.py
   ```
   Your web browser should open with the application running.

### Deploying to Streamlit Cloud:
For a live, shareable version of your application, you can deploy it to Streamlit Cloud for free.
1. **Push your code to a GitHub repository.**
2. **Sign up for Streamlit Cloud:** Go to [share.streamlit.io](https://share.streamlit.io/) and sign up with your GitHub account.
3. **Deploy the app:**
   - Click on "New app".
   - Select your repository and the branch where your code is.
   - Ensure the "Main file path" is set to `main.py`.
   - Click "Deploy!".

## Workshop Sessions 📚
Each session corresponds to a Jupyter notebook in the `soluciones/` directory.

| Session | Notebook | Topic |
|---|---|---|
| 1 | `01_load_and_clean_data.ipynb` | Loading, Inspecting & Cleaning Data |
| 2 | `02_transform_data.ipynb` | Data Transformation |
| 3 | `03_baseline_model.ipynb` | Creating a Baseline Model |
| 4 | `04_random_forest_model.ipynb` | Random Forest Classification |
| 5 | `05_xgboost_model.ipynb` | XGBoost Classification |
| 6 | `06_light_gbm_model.ipynb` | LightGBM (LGBM) Classification |
| 7 | `07_automl_pycaret.ipynb` | AutoML with PyCaret |
| (Optional) | `main.py` (Streamlit app) | Deploying a Model with Streamlit (Demo) |

## Learning Outcomes 🎯
By the end of this workshop, you’ll be able to:
* Confidently load, clean, and prepare data for machine learning tasks.
* Apply various data transformation techniques.
* Build, train, and evaluate several industry-standard classification models in Python.
* Understand the differences and trade-offs between Random Forest, XGBoost, and LightGBM.
* Use AutoML to automate the model selection process.
* Follow a structured approach to solving classification problems with machine learning.
* Deploy a machine learning model as a web application using Streamlit.

## Repository Structure 📁
```
.
├── figuras/                     # Contains images used in the Streamlit application
│   └── icone.png
│   └── maintenance.jpg
├── soluciones/                  # Workshop notebooks: from data processing to modeling
│   ├── 01_load_and_clean_data.ipynb
│   ├── 02_transform_data.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_random_forest_model.ipynb
│   ├── 05_xgboost_model.ipynb
│   ├── 06_light_gbm_model.ipynb
│   └── 07_automl_pycaret.ipynb
├── data/                        # Contains datasets for the workshop
│   ├── raw_data.csv
│   ├── clean_data.csv
│   └── transformed_data.csv
├── main.py                      # Example Streamlit application script for model deployment
├── model/                       # Contains a pre-trained example model and preprocessing objects
│   ├── final_model.joblib
│   ├── label_encoder.pkl
│   └── preprocessor_pipeline.pkl
├── requirements.txt             # Lists Python dependencies for local setup / Colab
└── utils.py                     # Utility functions for metrics and preprocessing
```

## Prerequisites 📾
* **Basic Python skills:** Familiarity with data types, loops, functions, and basic syntax.
* **Some knowledge of basic machine learning concepts:** Understanding of terms like features, target, training, testing, and model evaluation.
* **Familiarity with pandas and scikit-learn (optional):** Helpful, but not strictly required. The workshop will guide you through their usage.

## Additional Resources (Optional) 📚
* [Python Data Science Handbook by Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/)
* [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
* [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)
* [PyCaret Documentation](https://pycaret.org/)
* [Streamlit Documentation](https://docs.streamlit.io/)
