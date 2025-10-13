import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# --- Load dataset ---
iris = load_iris(as_frame=True)
df = iris.frame
df.columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','target']

target_map = {0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}
df['Species'] = df['target'].map(target_map)

X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train SVM models
kernels = ['linear', 'poly', 'rbf']
models = {}
for kernel in kernels:
    clf = SVC(kernel=kernel, probability=False, random_state=42)
    clf.fit(X, y_enc)
    models[kernel] = clf

# --- Streamlit UI ---
st.title("Iris Species Predictor (SVM)")
st.write("Enter feature values and choose a kernel to predict the Iris species.")

# User input
sl = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sw = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
pl = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
pw = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

kernel_choice = st.selectbox("Choose SVM Kernel", options=kernels)

sample = pd.DataFrame([[sl, sw, pl, pw]], columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])

if st.button("Predict"):
    clf = models[kernel_choice]
    pred = clf.predict(sample)[0]
    pred_label = le.inverse_transform([pred])[0]
    st.success(f"Predicted Species ({kernel_choice.capitalize()} Kernel): {pred_label}")
