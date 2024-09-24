import pandas as pd
import pandas as pd
import numpy as np
import joblib as jb
import streamlit as st
st.header("Restaurant Rating Prediction App")
st.caption("Welcome to my App")
dataset=pd.read_csv("Dataset.csv")
st.divider()
city = st.selectbox("Enter Name of the City:",np.sort(dataset["City"].unique()))
cft = st.number_input("Enter Average Cost of Two Dish",min_value=20,max_value=10000000,step=500)
htb = st.selectbox("Restaurant has Table Booking ?",["Yes","No"])
hod = st.selectbox("Restaurant has Online Delivery ?",["Yes","No"])
range = st.slider("What is the Price range (1 - Cheapest and 4 - Expensive)",4,1)
Button=st.button("Predict the Review!")
st.divider()

main_model=jb.load("pickle_model\main_model.pkl")
city_model=jb.load("pickle_model\city_model.pkl")
hod_model=jb.load("pickle_model\hod_model.pkl")
htb_model=jb.load("pickle_model\htb_model.pkl")
sc_model=jb.load("pickle_model\sc_model.pkl")

df=pd.DataFrame([[city,cft,htb,hod,range]],columns=['City', 'Average Cost for two', 'Has Table booking',
       'Has Online delivery', 'Price range'])
df["City"]=city_model.transform(df["City"])
df["Has Table booking"]=htb_model.transform(df["Has Table booking"])
df["Has Online delivery"]=hod_model.transform(df["Has Online delivery"])
df=sc_model.transform(df)
output = main_model.predict(df)

if Button:
    st.markdown("PREDICTING AGGREGATE RATINGS:" + str(output))
    if output <= 2.4 :
        st.write("Rating : Poor")
    elif output <= 3.4 :
        st.write("Rating : Average")
    elif output <= 3.9:
        st.write("Rating : Good")
    elif output <=4.4:
        st.write("Rating : Very Good")
    else:
        st.write("Rating : Excellent")

