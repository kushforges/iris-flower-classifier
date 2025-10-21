import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target_names[y]

# Streamlit app title
st.title("Iris Flower Classification")

# Welcome message
st.write("This app classifies iris flowers into three species based on sepal and petal dimensions.")
st.write("You can also experiment with different models and explore the dataset.")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox("Choose a model", ["Random Forest", "Logistic Regression", "K-Nearest Neighbors"])

# Select the model based on user input
if model_option == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=200)
else:
    model = KNeighborsClassifier()

# Train the model
model.fit(X, y)

# Show "How it works" section
if st.sidebar.checkbox('Show How It Works'):
    st.subheader("How It Works")
    st.write("""
    The model is trained using a well-known dataset called the Iris dataset, which contains measurements 
    of flowers and their corresponding species. We use these features (sepal and petal length/width) to train 
    a classifier that predicts the species of a flower based on the input values.
    """)

# Display dataset distribution (pairplot)
st.subheader("Iris Dataset Distribution")
sns.pairplot(df, hue='species')
st.pyplot(plt)

# Interactive Scatter Plot using Plotly
st.subheader("Interactive Data Visualization")
fig = px.scatter(df, x="sepal length (cm)", y="sepal width (cm)", color="species", title="Iris Dataset Visualization")
st.plotly_chart(fig)

# Input form for flower dimensions
st.subheader("Enter Flower Measurements")
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Classify Flower"):
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(input_data)
    species = iris.target_names[prediction[0]]
    
    # Display the result
    st.success(f"The predicted species is: {species}")

    # Display model performance metrics
    st.subheader("Model Performance Metrics")
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    st.text(f"Model Accuracy: {accuracy*100:.2f}%")
    st.text(f"Classification Report: \n{classification_report(y, y_pred)}")

# Show Feature Importance Bar Chart for Random Forest
if model_option == "Random Forest":
    st.subheader("Feature Importance (Random Forest)")
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': iris.feature_names,
        'Importance': feature_importance
    })
    st.bar_chart(importance_df.set_index('Feature'))

# Footer
st.write("""
### Next Steps:
- You can change the model in the sidebar to compare performance across different models.
- Upload your dataset to retrain the model.
- Visualize the Iris data and see how different features relate to each species.
""")
