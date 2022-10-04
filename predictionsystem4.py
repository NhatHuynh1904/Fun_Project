import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf



a = tf.keras.models.load_model("trained.sav")
x_test_data = pd.read_csv("/Users/huynhnhat/Desktop/Machine learning practice/Andrew Ng ml learning/Multiclassification Bean Class/Dry_Bean_Dataset_Test.csv", usecols=["Area",
                                                        "Perimeter",
                                                        "MajorAxisLength",
                                                        "MinorAxisLength",
                                                        "AspectRation", "Eccentricity",
                                                        "ConvexArea",
                                                        "EquivDiameter",
                                                        "Extent",
                                                        "Solidity",
                                                        "roundness",
                                                        "Compactness",
                                                        "ShapeFactor1",
                                                        "ShapeFactor2",
                                                        "ShapeFactor3",
                                                        "ShapeFactor4"])
y_test_data = np.array(pd.read_csv("/Users/huynhnhat/Desktop/Machine learning practice/Andrew Ng ml learning/Multiclassification Bean Class/Dry_Bean_Dataset_Test.csv", usecols= ["Class"]))
y_test_data = np.select([y_test_data == "SEKER", y_test_data == "BARBUNYA", y_test_data == "BOMBAY", y_test_data == "CALI"], [0, 1, 2, 3], y_test_data).astype(np.float32)

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




