import os
import requests
import pandas as pd
import streamlit as st

API = os.getenv("BACKEND_URL", "http://backend:7860")  # container name on the Docker network
st.title("SuperKart Sales Forecast")

st.subheader("Single prediction")
c1, c2 = st.columns(2)
with c1:
    weight = st.number_input("Product Weight", 0.0, 50.0, 12.66)
    sugar  = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    area   = st.number_input("Allocated Area", 0.0, 1.0, 0.027, format="%.3f")
    mrp    = st.number_input("Product MRP", 0.0, 500.0, 117.08)
    pid    = st.selectbox("Product Id prefix", ["FD", "DR", "NC"])
with c2:
    size  = st.selectbox("Store Size", ["Small", "Medium", "High"])
    city  = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
    stype = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2",
                                        "Departmental Store", "Food Mart"])
    age   = st.number_input("Store Age (years)", 0, 100, 16)
    cat   = st.selectbox("Product Category", ["Perishables", "Non Perishables"])

if st.button("Predict"):
    payload = {"Product_Weight": weight, "Product_Sugar_Content": sugar,
               "Product_Allocated_Area": area, "Product_MRP": mrp, "Store_Size": size,
               "Store_Location_City_Type": city, "Store_Type": stype, "Product_Id_char": pid,
               "Store_Age_Years": age, "Product_Type_Category": cat}
    r = requests.post(f"{API}/v1/predict", json=payload)
    if r.ok:
        st.success(f"Predicted sales: {r.json()['Predicted_Sales']}")
    else:
        st.error(r.text)

st.subheader("Batch prediction")
f = st.file_uploader("Upload CSV", type="csv")
if f is not None and st.button("Predict batch"):
    r = requests.post(f"{API}/v1/predictbatch", files={"file": f.getvalue()})
    if r.ok:
        st.dataframe(pd.DataFrame(r.json().items(), columns=["Row", "Predicted_Sales"]))
    else:
        st.error(r.text)
