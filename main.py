import joblib
import streamlit as st
import pandas as pd


# Page config
st.set_page_config(
    page_title="Failure Classifier",
    page_icon="images/icone.png",
)

# Page title
st.title('Maintenance - Failure Prediction')
st.image('images/maintenance.jpg')
st.write("\n\n")

st.markdown(
    """
    This app aims to assist in classifying failures, thereby reducing the time required to analyze machine problems. It enables the analysis of sensor data to classify failures swiftly and expedite the troubleshooting process.    
    """
)

# Load the saved encoder (to use lgbm model)
#encoder = joblib.load('C:/Users/mjkipsz2/OneDrive - The University of Manchester/Desktop/Pump failure/model/onehot_encoder.pkl')

# Load the model
model_file = 'C:/Users/mjkipsz2/OneDrive - The University of Manchester/Desktop/Pump failure/model/model.joblib'
model = joblib.load(model_file)

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
    # Transform the input using the encoder
    #type_encoded = encoder.transform([[type]])  # Ensure input is 2D
    #type_encoded_df = pd.DataFrame(type_encoded, columns=encoder.get_feature_names_out(['Type']))

# Function to predict the input
def prediction(air, process, rpm, torque, tool_wear, type):
    # Create a df with input data
    df_input = pd.DataFrame({
        'Air_temperature': [air],
        'Process_temperature': [process],
        'Rotational_speed': [rpm],
        'Torque': [torque],
        'Tool_wear': [tool_wear],
        'Type': [type]
    })

    prediction = model.predict(df_input)
    return prediction

# Botton to predict
if st.button('Predict'):
    predict = prediction(air, process, rpm, torque, tool_wear, type)
    st.success(predict)