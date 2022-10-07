import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf



a = tf.keras.models.load_model("trained.sav")


def prediction_bean(input_data):
    to_numpy = np.asarray(input_data)
    reshape_data = to_numpy.reshape(1, -1)
    prediction = a.predict(reshape_data)
    final = np.argmax(prediction)
    if final == 0:
        return "Seker"
    elif final == 1:
        return "Barbunya"
    elif final == 2:
        return "Bombay"
    elif final == 3:
        return "Cali"
    elif final == 4:
        return "Dermosan"
    elif final == 5:
        return "Horoz"
    elif final == 6: 
        return "Sira"

  


def main():
    st.title("Bean prediction Web App")

    #Get input data:
    Area = st.number_input("Area of bean? (20.000 - 300.000)")
    Perimeter = st.number_input("Perimeter? 500-2000")
    MajorAxisLength = st.number_input("Major Axis 200-500")
    MinorAxisLength = st.number_input("Minor Axis 100-400")
    AspectRation = st.number_input("AspectRation 1-3")
    Eccentricity = st.number_input("Eccentricity 0.5 - 0.9")
    ConvexArea = st.number_input("ConvexArea 20000 - 200000")
    EquivDiameter = st.number_input("EquivDiameter 150 - 300")
    Extent = st.number_input(" Extent 0.4 - 0.9")
    Solidity = st.number_input(" Solidity 0.8 - 1")
    Roundness = st.number_input(" roundness 0.6 - 1")
    Compactness = st.number_input("Compactness 0.6 - 1")
    ShapeFactor1 = st.number_input("ShapeFactor1 0.005 - 0.008")
    ShapeFactor2 = st.number_input("ShapeFactor2 0.001 - 0.003")
    ShapeFactor3 = st.number_input("ShapeFactor3 0.4 - 0.9")
    ShapeFactor4 = st.number_input("ShapeFactor4 0.9 - 1")

    type = 0
    if st.button("Type of bean"):
        type = prediction_bean([Area, Perimeter,  MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent,  Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4])

    st.success(type)

if __name__ == '__main__':
    main()




