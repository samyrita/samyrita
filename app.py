import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScalerModel, PCAModel
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

# Create a Spark session (create it once)
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# --- Load Your Spark Models (Replace with actual paths) ---
scaler_model_path = "/content/scaler_model" 
pca_model_path = "/content/pca_model"      
model_path = "/content/regression_model"           

try:
    scaler_model = StandardScalerModel.load(scaler_model_path)
    pca_model = PCAModel.load(pca_model_path)
    model = RandomForestRegressionModel.load(model_path)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop() 

st.title("Wine Quality Prediction")

# Input fields for wine characteristics
fixed_acidity = st.number_input("Fixed Acidity", value=7.0)
volatile_acidity = st.number_input("Volatile Acidity", value=0.27)
citric_acid = st.number_input("Citric Acid", value=0.36)
residual_sugar = st.number_input("Residual Sugar", value=20.7)
chlorides = st.number_input("Chlorides", value=0.045)
density = st.number_input("Density", value=0.9978)
alcohol = st.number_input("Alcohol", value=8.8)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=45.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=170.0)

# Predict button
if st.button("Predict"):
    # Create a Spark DataFrame from the input values
    input_data = spark.createDataFrame([
        Row(fixed_acidity=fixed_acidity, volatile_acidity=volatile_acidity, 
citric_acid=citric_acid, residual_sugar=residual_sugar,
            chlorides=chlorides, density=density,
            alcohol=alcohol,
            free_sulfur_dioxide=free_sulfur_dioxide,
            total_sulfur_dioxide=total_sulfur_dioxide)
    ])

# Assemble features
    assembler = VectorAssembler(inputCols=["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                                            "chlorides", "density", "alcohol", "free_sulfur_dioxide", "total_sulfur_dioxide"],
                                outputCol="assembled_features")
    input_data_assembled = assembler.transform(input_data)

    # Scale features
    scaled_input_data = scaler_model.transform(input_data_assembled)

    # Apply PCA
    pca_input_data = pca_model.transform(scaled_input_data)

    # Make prediction
    prediction = model.transform(pca_input_data)

    # Display prediction
    predicted_quality = prediction.select("prediction").collect()[0][0]
    st.success(f"Predicted Wine Quality: {predicted_quality:.2f}")
