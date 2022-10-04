import streamlit as st
import numpy as np
import joblib

a = joblib.load("/Train_Bean.sav")
print(a)

def prediction_bean(input_data):
    to_numpy = np.asarray(input_data)
    reshape_data = to_numpy.reshape(1, -1)
    prediction = a.predict(reshape_data)
    final = np.argmax(prediction)
    if final == 0:
        return "Work!"


def main():
    st.title("Bean prediction Web App")

    #Get input data:
    Area = st.number_input("Area of bean?")
    Perimeter = st.number_input("Perimeter?")
    MajorAxisLength = st.number_input("Major Axis")
    MinorAxisLength = st.number_input("Minor Axis")
    AspectRation = st.number_input("AspectRation")
    Eccentricity = st.number_input("Eccentricity")
    ConvexArea = st.number_input(" ConvexArea")
    EquivDiameter = st.number_input("EquivDiameter")
    Extent = st.number_input(" Extent")
    Solidity = st.number_input(" Solidity")
    Roundness = st.number_input(" roundness ")
    Compactness = st.number_input("Compactness")
    ShapeFactor1 = st.number_input("ShapeFactor1")
    ShapeFactor2 = st.number_input("ShapeFactor2")
    ShapeFactor3 = st.number_input("ShapeFactor3")
    ShapeFactor4 = st.number_input("ShapeFactor4 ")

    type = 0
    if st.button("Type of bean"):
        type = prediction_bean([Area, Perimeter,  MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent,  Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4])

    st.success(type)

if __name__ == '__main__':
    main()




