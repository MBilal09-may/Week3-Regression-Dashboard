import pandas as pd
import streamlit as st
import joblib
from src.predict import predict_price
import matplotlib.pyplot as plt
from src.preprocessing import load_data, cleaned_df


#Loading saved files
encoder = joblib.load("models/encoder.pkl")
scaler = joblib.load("models/scaler.pkl")
best_model = joblib.load("models/best_model.pkl")
metrics = joblib.load("models/best_metrics.pkl")



df = pd.read_csv("dataset.csv")
df = df.dropna()
df = load_data()
df = cleaned_df((df))

addresses = df["Address"].unique()

#Page title
def title():
    st.header("Welcome")
    st.title("🏠 House Price Predictor")
    st.markdown(
    "Enter the house details below to get an estimated price based on our trained machine learning model."
)
# function for Widget for area
def area():
    user_area= st.number_input(
        "Enter area (m²)",
        min_value=200,
        step=100
    )
    return user_area
#Function for widget of rooms
def room():
    rooms = st.number_input(
        "Enter number of rooms",
        min_value=1
    )
    return rooms
#functions for widget of parking,warehouse and elevator

def parking():
    parking_value = int(st.checkbox("Parking"))
    return parking_value
def warehouse():
    warehouse_value = int(st.checkbox("Warehouse"))
    return warehouse_value
def elevator():
    elevator_value = int(st.checkbox("Elevator"))
    return elevator_value


def address():
    selected_addresses = st.selectbox(
        "Choose address",
        addresses
    )
    return selected_addresses


#Calling all the functions
title()
user_area = area()
user_rooms = room()
user_parking = parking()
user_warehouse = warehouse()
user_elevator = elevator()
user_address = address()




if st.button("Predict Prize"):
    input_data = pd.DataFrame({
        "Area":[user_area],
        "Room":[user_rooms],
        "Parking":[user_parking],
        "Warehouse":[user_warehouse],
        "Elevator":[user_elevator],
        "Address":[user_address]
    })
    prediction = predict_price(input_data)


 

    st.write(f"💰 Estimated Price")
    st.write(prediction)

st.subheader("📊 Model Performance")

st.metric(
    "R² Score",
    round(metrics["R2 score"],3)
)

st.metric(
    "MAE",
    round(metrics["MAE"], 2)
)

st.metric(
    "MSE",
    round(metrics["MSE"], 2)
)

st.subheader("📈 Charts")

#Bar
chart_data = df.groupby("Room")["Price"].mean()
fig, ax = plt.subplots()
chart_data.plot(kind="bar",ax=ax)
st.pyplot(fig)

# Scatter plot

fig, ax = plt.subplots()

ax.scatter(df["Area"], df["Price"])
st.pyplot(fig)

#Histogram

fig, ax = plt.subplots()

ax.hist(df["Price"], bins=10)

st.pyplot(fig)

st.subheader("📌 Dataset Insights")


#Average house price
st.write(f"Total houses: {df.shape[0]}")
st.write(f"Average price: {round(df['Price'].mean(),0)}")
st.write(f"Average area: {round(df['Area'].mean(),0)}")


#Most expensive area
expensive_areas = (
    df.groupby("Address")["Price"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

st.write("Top expensive areas")
st.write(expensive_areas)


#Average room price
room_price = df.groupby("Room")["Price"].mean()

st.bar_chart(room_price)


