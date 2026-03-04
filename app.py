import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

st.title("ASG 04 MD - Bevlin Logen - Spaceship Titanic Model Deployment")
st.write("Input Passenger Features To Predict Whether They Will Be Transported")

homeplanet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
cryosleep = st.selectbox("CryoSleep", [True, False])
destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
age = st.number_input("Age", 0, 100, 30)

vip = st.selectbox("VIP", [True, False])

roomservice = st.number_input("RoomService", 0.0, 10000.0, 0.0)
foodcourt = st.number_input("FoodCourt", 0.0, 10000.0, 0.0)
shoppingmall = st.number_input("ShoppingMall", 0.0, 10000.0, 0.0)
spa = st.number_input("Spa", 0.0, 10000.0, 0.0)
vrdeck = st.number_input("VRDeck", 0.0, 10000.0, 0.0)

deck = st.selectbox("Cabin Deck", ["A", "B", "C", "D", "E", "F", "G"])
side = st.selectbox("Cabin Side", ["P", "S"])

input_data = pd.DataFrame({
    "PassengerId": ["0001_01"],
    "HomePlanet": [homeplanet],
    "CryoSleep": [cryosleep],
    "Cabin": ["F/1234/P"],
    "Destination": [destination],
    "Age": [age],
    "VIP": [vip],
    "RoomService": [roomservice],
    "FoodCourt": [foodcourt],
    "ShoppingMall": [shoppingmall],
    "Spa": [spa],
    "VRDeck": [vrdeck],
    "Name": ["John Doe"],
    "CabinDeck": [deck],
    "CabinSide": [side]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    
    if prediction[0] == True:
        st.success("Passenger Was Transported")
    else:
        st.error("Passenger Was Not Transported")