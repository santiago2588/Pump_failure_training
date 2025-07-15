import joblib
import streamlit as st
import pandas as pd
import altair as alt

# --- Page Configuration ---
# Use an absolute path or make sure the icon is in the correct relative path
try:
    st.set_page_config(
        page_title="Failure Classifier",
        page_icon="figuras/icone.png",
        layout="wide"
    )
except FileNotFoundError:
     st.set_page_config(
        page_title="Failure Classifier",
        layout="wide"
    )


# --- Load Models and Preprocessing Objects ---
# This is a cached function to prevent reloading the models on every interaction
@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load('model/preprocessor_pipeline.pkl')
        model = joblib.load('model/final_model.joblib')
        label_encoder = joblib.load('model/label_encoder.pkl')
        return preprocessor, model, label_encoder
    except FileNotFoundError as e:
        st.error(
            f"Error loading model files: {e}. "
            "Please make sure 'preprocessor_pipeline.pkl', 'final_model.joblib', and 'label_encoder.pkl' "
            "are in the 'model/' directory."
        )
        return None, None, None
    except AttributeError as e:
        st.error(
            f"AttributeError: {e}\n\n"
            "This usually means there is a version mismatch for scikit-learn "
            "between your training environment and this Streamlit environment. "
            "Please create a requirements.txt file and specify the exact version "
            "of scikit-learn used for training (e.g., scikit-learn==1.2.2)."
        )
        return None, None, None

preprocessor, model, label_encoder = load_models()

# --- Mappings and Descriptions ---
FAILURE_DESCRIPTIONS = {
    'No Failure': "✅ The machine is operating under normal conditions. No immediate maintenance is required.",
    'Heat Dissipation Failure': "🔥 The machine is overheating. This could be due to issues with the cooling system, high ambient temperatures, or prolonged high-torque operation. Check for blockages in air vents and ensure the cooling system is functional.",
    'Power Failure': "⚡ The model predicts a potential power failure. This is often linked to sudden drops in torque or inconsistencies in rotational speed without corresponding tool wear. Check power supply and electrical connections.",
    'Overstrain Failure': "⚙️ The machine is under excessive strain, indicated by high torque combined with low rotational speed. This can lead to component damage. Reduce the workload or check for mechanical obstructions.",
    'Tool Wear Failure': "🔧 The tool is significantly worn and needs replacement. This is a common type of failure and is directly indicated by the 'Tool wear' metric."
}


# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Input Parameters")

air_input = st.sidebar.slider('Air Temperature [K]', min_value=290.0, max_value=310.0, value=300.0, step=0.1)
process_input = st.sidebar.slider('Process Temperature [K]', min_value=300.0, max_value=320.0, value=310.0, step=0.1)
rpm_input = st.sidebar.slider('Rotational Speed [rpm]', min_value=1100, max_value=3000, value=1500, step=10)
torque_input = st.sidebar.slider('Torque [Nm]', min_value=3.0, max_value=80.0, value=40.0, step=0.1)
tool_wear_input = st.sidebar.slider('Tool Wear [min]', min_value=0, max_value=260, value=100, step=1)
type_input = st.sidebar.selectbox('Machine Quality Type', options=['Low', 'Medium', 'High'])


# --- Main App Interface ---
st.title('🛠️ Predictive Maintenance: Failure Classifier')
st.markdown(
    """
    This application uses a machine learning model to predict potential equipment failures based on real-time sensor data.
    Adjust the parameters in the sidebar to see how they affect the failure prediction.
    """
)
st.write("---")

# --- Prediction Function ---
def prediction(air_temp, proc_temp, rotational_speed, torque_val, tool_wear_val, type_val):
    if not all([preprocessor, model, label_encoder]):
        return None, None # Models not loaded

    input_data = {
        'Air_temperature': [air_temp],
        'Process_temperature': [proc_temp],
        'Rotational_speed': [rotational_speed],
        'Torque': [torque_val],
        'Tool_wear': [tool_wear_val],
        'Type': [type_val]
    }
    df_input = pd.DataFrame(input_data)
    df_transformed = preprocessor.transform(df_input)
    prediction_proba = model.predict_proba(df_transformed)
    prediction_class = model.predict(df_transformed)
    return prediction_class[0], prediction_proba

# --- Prediction Execution and Output ---
if st.sidebar.button('▶️ Predict Failure Type', type="primary"):
    predicted_class_num, prediction_confidence = prediction(
        air_temp=air_input,
        proc_temp=process_input,
        rotational_speed=rpm_input,
        torque_val=torque_input,
        tool_wear_val=tool_wear_input,
        type_val=type_input
    )

    if predicted_class_num is not None:
        predicted_label = label_encoder.inverse_transform([predicted_class_num])[0]

        st.header("Prediction Result")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(label="Predicted Failure", value=predicted_label)
            st.write("**Description:**")
            st.info(FAILURE_DESCRIPTIONS.get(predicted_label, "No description available."))

        with col2:
            st.write("**Prediction Confidence**")
            confidence_df = pd.DataFrame(prediction_confidence, columns=label_encoder.classes_).T
            confidence_df = confidence_df.reset_index()
            confidence_df.columns = ['Failure Type', 'Confidence']

            chart = alt.Chart(confidence_df).mark_bar().encode(
                x=alt.X('Confidence:Q', axis=alt.Axis(format='%')),
                y=alt.Y('Failure Type:N', sort='-x'),
                tooltip=['Failure Type', alt.Tooltip('Confidence:Q', format='.2%')]
            ).properties(
                title='Model Confidence for Each Failure Type'
            )
            st.altair_chart(chart, use_container_width=True)

else:
    st.info("Adjust the parameters in the sidebar and click 'Predict Failure Type' to see a result.")

# --- Explainer Section ---
with st.expander("ℹ️ About the Application"):
    st.markdown("""
    **How does it work?**

    1.  **Input Data:** You provide the current sensor readings from the machine using the sliders and dropdown on the left.
    2.  **Preprocessing:** The app takes your raw inputs and transforms them (scaling numerical values and encoding categorical ones) so the model can understand them.
    3.  **Prediction:** The pre-trained LightGBM (LGBM) model analyzes the transformed data and calculates the probability of each potential failure type.
    4.  **Output:** The app displays the most likely failure type and a chart showing the model's confidence in each prediction.

    **Model Details:**

    * **Model Type:** `LightGBM Classifier`
    * **Purpose:** To classify different types of machine failures based on sensor data.
    * **Features Used:** Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear, and Machine Quality Type.
    """)

