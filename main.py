import joblib
import streamlit as st
import pandas as pd


# Page config
st.set_page_config(
    page_title="Failure Classifier",
    page_icon="figuras/icone.png",
)

# Page title
st.title('Maintenance - Failure Prediction')
st.image('figuras/maintenance.jpg')
st.write("\n\n")

st.markdown(
    """
    This app aims to assist in classifying failures, thereby reducing the time required to analyze machine problems. It enables the analysis of sensor data to classify failures swiftly and expedite the troubleshooting process.    
    """
)

# Load the saved encoder (to use lgbm model)
preprocessor = joblib.load('model/preprocessor_pipeline.pkl')

# Load the model
model_file = 'model/final_model.joblib'
model = joblib.load(model_file)

# --- User Input Interface ---
st.header("Input Sensor Data")

# Streamlit interface to input data
col1, col2 = st.columns(2)

with col1:
    air = st.slider(label='Air Temperature [K]',min_value=250, max_value=350, value=300, step=1)
    process = st.slider(label='Process Temperature [K]',min_value=300, max_value=350, value=310, step=1)
    rpm = st.slider(label='Rotational Speed [rpm]',min_value=1000, max_value=3000, value=1500, step=50)

with col2:
    torque = st.slider(label='Torque [Nm]',min_value=1, max_value=100, value=40, step=1)
    tool_wear = st.slider(label='Tool Wear [min]',min_value=0, max_value=300, value=100, step=5)
    type = st.selectbox(label='Type', options=['Low', 'Medium', 'High'])

# Function to predict the input
def prediction(air_temp, proc_temp, rotational_speed, torque_val, tool_wear_val, type_val):
    """
    This function takes raw user inputs, preprocesses them using the pipeline,
    and returns a model prediction.
    """
    # 1. Create a DataFrame from the inputs with the correct column names
    # The column names must exactly match those used when the preprocessor was trained
    input_data = {
        'Air_temperature': [air_temp],
        'Process_temperature': [proc_temp],
        'Rotational_speed': [rotational_speed],
        'Torque': [torque_val],
        'Tool_wear': [tool_wear_val],
        'Type': [type_val]
    }
    df_input = pd.DataFrame(input_data)

    # 2. Apply the entire preprocessing pipeline
    df_transformed = preprocessor.transform(df_input)

    # 3. Make the prediction
    # Use predict_proba to get the confidence score
    prediction_proba = model.predict_proba(df_transformed)
    prediction_class = model.predict(df_transformed)

    return prediction_class[0], prediction_proba

# --- Prediction Button and Output ---
if st.button('Predict Failure Type', type="primary"):
    # Call the prediction function with the user inputs
    predicted_class, prediction_confidence = prediction(air, process, rpm, torque, tool_wear, type_input)

    st.write("---")
    st.header("Prediction Result")

    # Display the result with confidence
    st.success(f"The predicted failure type is: **{predicted_class}**")

    # Display the probabilities for each class
    st.write("Prediction Confidence:")
    confidence_df = pd.DataFrame(prediction_confidence, columns=model.classes_)
    st.dataframe(confidence_df)
