import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import time


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

# app.py is inside:
# Project/
#     dashboards/
#         app.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "Data" / "processed" / "featured_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"


# ============================================================
# TITLE
# ============================================================

st.title("⚙️ Industrial Predictive Maintenance System")

st.write(
    "Continuous machine-data monitoring and machine failure prediction"
)


# ============================================================
# CHECK FILES
# ============================================================

if not DATA_PATH.exists():

    st.error(
        f"Feature engineering data was not found:\n\n{DATA_PATH}"
    )

    st.stop()


if not MODEL_PATH.exists():

    st.error(
        f"Trained model was not found:\n\n{MODEL_PATH}"
    )

    st.info(
        "Run model_preparation.py first and save final_model.joblib."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    return data


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    trained_model = joblib.load(MODEL_PATH)

    return trained_model


df = load_data()
model = load_model()


# ============================================================
# REQUIRED MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "Air_Temp_K",
    "Process_Temp_K",
    " Rotational_Speed_RPM",
    "Torque_Nm",
    "Tool_Wear_Min",
    "temperature_diff",
    "Tool_Wear_type",
    "RPM_type",
    "Torque_Category",
    "Total_failure_count",
    "High_Risk_flag"
]


# ============================================================
# CHECK MODEL FEATURES IN DATA
# ============================================================

missing_columns = [
    column
    for column in MODEL_FEATURES
    if column not in df.columns
]


if missing_columns:

    st.error("Required model columns are missing from the CSV:")

    for column in missing_columns:
        st.write(f"- {column}")

    st.write("Available columns:")
    st.write(df.columns.tolist())

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Monitoring Controls")


monitoring_mode = st.sidebar.radio(
    "Select Mode",
    [
        "Dashboard Monitoring",
        "Manual Prediction"
    ]
)


# ============================================================
# AUTOMATIC DASHBOARD MONITORING
# ============================================================

if monitoring_mode == "Dashboard Monitoring":

    st.header("📡 Continuous Machine Monitoring")

    st.write(
        "Machine data is read sequentially from the feature-engineered "
        "dataset and passed automatically to the trained model."
    )


    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        start_monitoring = st.button(
            "▶ Start Monitoring"
        )

    with col2:

        stop_monitoring = st.button(
            "⏹ Stop Monitoring"
        )


    if "monitoring" not in st.session_state:

        st.session_state.monitoring = False


    if start_monitoring:

        st.session_state.monitoring = True


    if stop_monitoring:

        st.session_state.monitoring = False


    # --------------------------------------------------------
    # INITIAL DASHBOARD
    # --------------------------------------------------------

    if not st.session_state.monitoring:

        # Predict entire dataset for dashboard summary

        X_all = df[MODEL_FEATURES]

        predictions = model.predict(X_all)

        probabilities = model.predict_proba(X_all)[:, 1]


        healthy_count = int((predictions == 0).sum())

        failure_count = int((predictions == 1).sum())


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        m1, m2, m3 = st.columns(3)


        with m1:

            st.metric(
                "Total Machines",
                len(df)
            )


        with m2:

            st.metric(
                "Healthy Machines",
                healthy_count
            )


        with m3:

            st.metric(
                "Predicted Failures",
                failure_count
            )


        st.subheader("📋 Machine Prediction Overview")


        display_df = df.copy()

        display_df["Predicted_Failure"] = predictions

        display_df["Failure_Probability"] = (
            probabilities * 100
        ).round(2)


        display_df["Health_Status"] = np.where(
            predictions == 1,
            "🔴 FAILURE RISK",
            "🟢 HEALTHY"
        )


        columns_to_show = []

        # Keep useful identification columns if available

        for column in [
            "UDI",
            "Product_ID",
            "Type",
            "Machine_ID"
        ]:

            if column in display_df.columns:

                columns_to_show.append(column)


        columns_to_show += [
            "Air_Temp_K",
            "Process_Temp_K",
            " Rotational_Speed_RPM",
            "Torque_Nm",
            "Tool_Wear_Min",
            "Predicted_Failure",
            "Failure_Probability",
            "Health_Status"
        ]


        st.dataframe(
            display_df[columns_to_show],
            use_container_width=True
        )


    # --------------------------------------------------------
    # CONTINUOUS SIMULATION
    # --------------------------------------------------------

    else:

        st.info(
            "Monitoring is running. Data is being processed sequentially."
        )


        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        machine_placeholder = st.empty()


        for index in range(len(df)):

            if not st.session_state.monitoring:

                break


            current_row = df.iloc[[index]]


            X_current = current_row[MODEL_FEATURES]


            prediction = int(
                model.predict(X_current)[0]
            )


            probability = float(
                model.predict_proba(X_current)[0][1]
            )


            # ------------------------------------------------
            # MACHINE STATUS
            # ------------------------------------------------

            if prediction == 1:

                status_placeholder.error(
                    f"🔴 MACHINE FAILURE RISK — Record {index + 1}"
                )

            else:

                status_placeholder.success(
                    f"🟢 MACHINE HEALTHY — Record {index + 1}"
                )


            # ------------------------------------------------
            # CURRENT MACHINE VALUES
            # ------------------------------------------------

            current_col1, current_col2, current_col3, current_col4 = (
                machine_placeholder.columns(4)
            )


            with current_col1:

                st.metric(
                    "Temperature",
                    f"{current_row['Process_Temp_K'].iloc[0]:.2f} K"
                )


            with current_col2:

                st.metric(
                    "Rotational Speed",
                    f"{current_row[' Rotational_Speed_RPM'].iloc[0]:.0f} RPM"
                )


            with current_col3:

                st.metric(
                    "Torque",
                    f"{current_row['Torque_Nm'].iloc[0]:.2f} Nm"
                )


            with current_col4:

                st.metric(
                    "Failure Probability",
                    f"{probability * 100:.2f}%"
                )


            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            progress = (index + 1) / len(df)

            st.progress(
                progress
            )


            time.sleep(1)


        st.success(
            "Monitoring session completed."
        )


# ============================================================
# MANUAL MACHINE PREDICTION
# ============================================================

else:

    st.header("🔎 Manual Machine Health Prediction")

    st.write(
        "Enter machine sensor and engineered feature values "
        "to evaluate the machine using the trained model."
    )


    # --------------------------------------------------------
    # NUMERICAL INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        air_temp = st.number_input(
            "Air Temperature (K)",
            value=float(df["Air_Temp_K"].median())
        )


        process_temp = st.number_input(
            "Process Temperature (K)",
            value=float(df["Process_Temp_K"].median())
        )


        rotational_speed = st.number_input(
            " Rotational Speed (RPM)",
            value=float(df[" Rotational_Speed_RPM"].median())
        )


        torque = st.number_input(
            "Torque (Nm)",
            value=float(df["Torque_Nm"].median())
        )


    with col2:

        tool_wear = st.number_input(
            "Tool Wear (min)",
            value=float(df["Tool_Wear_Min"].median())
        )


        total_failure_count = st.number_input(
            "Total Failure Count",
            min_value=0,
            value=int(df["Total_failure_count"].median())
        )


        temperature_diff = st.number_input(
            "Temperature Difference",
            value=float(df["temperature_diff"].median())
        )


    # --------------------------------------------------------
    # CATEGORICAL INPUTS
    # --------------------------------------------------------

    col3, col4 = st.columns(2)


    with col3:

        tool_wear_type = st.selectbox(
            "Tool Wear Type",
            sorted(
                df["Tool_Wear_type"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


        rpm_type = st.selectbox(
            "RPM Type",
            sorted(
                df["RPM_type"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


    with col4:

        torque_category = st.selectbox(
            "Torque Category",
            sorted(
                df["Torque_Category"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


        high_risk_flag = st.selectbox(
            "High Risk Flag",
            sorted(
                df["High_Risk_flag"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "Air_Temp_K": [air_temp],

        "Process_Temp_K": [process_temp],

        " Rotational_Speed_RPM": [rotational_speed],

        "Torque_Nm": [torque],

        "Tool_Wear_Min": [tool_wear],

        "temperature_diff": [temperature_diff],

        "Tool_Wear_type": [tool_wear_type],

        "RPM_type": [rpm_type],

        "Torque_Category": [torque_category],

        "Total_failure_count": [total_failure_count],

        "High_Risk_flag": [high_risk_flag]
    })


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Machine Health",
        type="primary"
    ):

        try:

            prediction = int(
                model.predict(input_data)[0]
            )


            probability = float(
                model.predict_proba(input_data)[0][1]
            )


            st.subheader("Prediction Result")


            if prediction == 1:

                st.error(
                    "🔴 MACHINE FAILURE RISK DETECTED"
                )

            else:

                st.success(
                    "🟢 MACHINE IS PREDICTED HEALTHY"
                )


            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            st.metric(
                "Failure Probability",
                f"{probability * 100:.2f}%"
            )


            st.progress(
                probability
            )


            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            st.subheader(
                "Machine Input Data"
            )


            st.dataframe(
                input_data,
                use_container_width=True
            )


        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )

            st.write(
                "Input columns received:"
            )

            st.write(
                input_data.columns.tolist()
            )
            