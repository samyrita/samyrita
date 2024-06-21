import streamlit as st
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScalerModel, PCAModel
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

# Create a Spark session (if not already running)
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# --- Load Your Spark Models (Replace with actual paths) ---
try:
    scaler_model = StandardScalerModel.load("/path/to/your/scaler_model")
    pca_model = PCAModel.load("/path/to/your/pca_model")
    model = RandomForestRegressionModel.load("/path/to/your/rf_model")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()  # Stop the app if models can't be loaded

st.title("Wine Quality Prediction")

# Input fields for wine characteristics
fixed_acidity = st.number_input("Fixed Acidity", value=7.0)
volatile_acidity = st.number_input("Volatile Acidity", value=0.5)
citric_acid = st.number_input("Citric Acid", value=0.3)
residual_sugar = st.number_input("Residual Sugar", value=2.0)
chlorides = st.number_input("Chlorides", value=0.07)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=15.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=50.0)
density = st.number_input("Density", value=0.995)
pH = st.number_input("pH", value=3.3)
sulphates = st.number_input("Sulphates", value=0.6)
alcohol = st.number_input("Alcohol", value=10.0)

# Create a feature vector from user input
input_features = np.array([fixed_acidity, volatile_acidity, citric_acid, 
                           residual_sugar, chlorides, free_sulfur_dioxide, 
                           total_sulfur_dioxide, density, pH, sulphates, 
                           alcohol]).reshape(1, -1)

# --- Prediction Logic ---
feature_cols = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", 
                "density", "pH", "sulphates", "alcohol"]
input_list = input_features.tolist()
input_df = spark.createDataFrame(input_list, schema=feature_cols)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
input_df = assembler.transform(input_df)

input_df = scaler_model.transform(input_df)
input_df = pca_model.transform(input_df)

prediction = model.transform(input_df).select("prediction").collect()[0][0]

# --- Display Prediction ---
st.subheader("Predicted Wine Quality:")
st.write(prediction)

#Debug de la app
!streamlit run /content/wineapp.py 
