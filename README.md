# Energy Forecasting with RNN and LSTM

This project explores the differences between **Recurrent Neural Networks (RNN)** and **Long Short-Term Memory (LSTM)** models for time series prediction. Using the **Energy Consumption Generation Prices and Weather** dataset from Kaggle, the goal is to predict energy generation from fossil gas using a 48-hour window to forecast the next 24 hours.


## **Project Objective**

1. Compare the performance of RNN and LSTM models in time series forecasting.
2. Evaluate the impact of preprocessing techniques: Original Data, MinMax Scaling, and Standardization.
3. Analyze model performance based on different metrics and loss functions.


## **Dataset**

The dataset contains information on:
- **Features**: Day of the week, Day of the month, Day of the year, Weekday/Weekend, Hour of the day, Weather information (e.g., temperature, humidity, wind speed).
- **Target**: `generation fossil gas` (energy generation from fossil gas).

Dataset source: [Energy Consumption Generation Prices and Weather on Kaggle](https://www.kaggle.com/datasets).


## **Methodology**

### **1. Preprocessing**
- Extracted time-based features: day, month, year, hour, weekend indicator.
- Handled missing values by replacing them with column means.
- Applied three preprocessing techniques:
  - Original data (no scaling).
  - **MinMax Scaling**: Normalizes data to the range [0, 1].
  - **Standardization**: Scales data to have zero mean and unit variance.

### **2. Models**
- Implemented **RNN** and **LSTM** models using TensorFlow and Keras.
- Models were trained to predict the next 24 hours using a 48-hour input window.

### **3. Loss Functions**
- Evaluated using three loss functions:
  - **Mean Squared Error (MSE)**.
  - **Mean Absolute Error (MAE)**.
  - **Mean Absolute Percentage Error (MAPE)**.

### **4. Evaluation Metrics**
- Model performance was assessed using:
  - MSE
  - MAE
  - MAPE
  - Symmetric MAPE (SMAPE)
  - R² (Coefficient of Determination)
- Running time was measured for each configuration.


## **Results**

### **Key Findings**
1. **Impact of Preprocessing**:
   - Standardization yielded the best results for both models.
   - MinMax Scaling improved model performance but was less effective than standardization.
   - Models performed poorly with unscaled original data.

2. **RNN vs. LSTM**:
   - **LSTM** consistently outperformed RNN in terms of accuracy (lower MSE, MAE, higher R²).
   - **RNN** was faster but less accurate, especially with longer dependencies in time series.

3. **Best Configuration**:
   - LSTM with Standardization achieved the best performance metrics:
     - Lowest MSE and MAE.
     - Highest R² value.


## **Technologies Used**

- **Python**: Core programming language.
- **Pandas**: Data preprocessing and feature extraction.
- **NumPy**: Numerical computations.
- **Matplotlib & Seaborn**: Visualizations.
- **Scikit-learn**: Preprocessing techniques and evaluation metrics.
- **TensorFlow/Keras**: Model development and training.
