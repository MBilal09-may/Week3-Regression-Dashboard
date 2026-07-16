# House Price Prediction Dashboard

## Project Title

House Price Prediction Dashboard using Machine Learning and Streamlit

---

## Objective

The objective of this project is to build a machine learning regression model that can predict house prices based on different house features.

The project focuses on creating a complete machine learning workflow, including data cleaning, feature preprocessing, model training, model evaluation, and connecting the final model with a Streamlit dashboard for user interaction.

---

## Dataset Description

The dataset used in this project contains information about houses and their features that affect the price.

The main features include:

* Area
* Number of rooms
* Parking availability
* Warehouse availability
* Elevator availability
* Address
* House price

The target variable for prediction is the house price.

The dataset was used to train and test different regression models to find the model that performs best.

---

## Data Preprocessing Steps

Before training the models, several preprocessing steps were performed:

* Removed unnecessary columns from the dataset
* Handled missing values
* Converted area values from string format into numerical format
* Converted binary features such as parking, warehouse, and elevator into numerical values
* Applied One Hot Encoding to the address column
* Applied feature scaling using StandardScaler
* Split the dataset into training and testing data

The same preprocessing steps were applied to user input data before making predictions in the Streamlit application.

---

## Regression Models Used

The following regression models were trained and compared:

### Linear Regression

Used as a basic regression model to understand the relationship between input features and house prices.

### Decision Tree Regressor

Used to capture more complex patterns in the dataset.

### Random Forest Regressor

Used as an ensemble model that combines multiple decision trees to improve prediction performance.

---

## Model Comparison

The models were evaluated using regression performance metrics:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

The models were compared based on their performance on the testing dataset.

The model with the highest R² score and better overall performance was selected as the final model.

---

## Best Model Selection

After comparing all trained models, the best performing model was saved using Joblib.

The saved files include:

* Trained regression model
* One Hot Encoder
* Feature scaler

These files are loaded in the Streamlit application to make predictions using new user input.

---

## Technologies Used

The following technologies and libraries were used:

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn
* Joblib

---

## Installation Guide

Clone the repository:

```bash
git clone <repository-link>
```

Move into the project folder:

```bash
cd week3-regression-dashboard
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Streamlit Execution Instructions

To run the dashboard, use:

    bash
streamlit run app.py


After running the command, the Streamlit application will open in the browser.

Users can enter house details and get an estimated house price prediction.



## Dashboard Screenshots

Screenshots of the Streamlit dashboard are included in the repository.

The screenshots show:

* Home page
* User input section
* Prediction results
* Model performance metrics
* Data visualizations
* Dataset insights

---

## GitHub Repository Link

Repository Link:

#Github link

https://github.com/MBilal09-may/Week3-Regression-Dashboard
---

## Conclusion

This project helped me understand the complete machine learning workflow, from preparing raw data to deploying a trained regression model through a Streamlit dashboard.

During development, I faced issues related to preprocessing, file paths, model saving/loading, and connecting different Python files. Solving these problems improved my understanding of how machine learning projects are structured in real-world applications.
