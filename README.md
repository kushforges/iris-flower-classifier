# 🌸 Iris Flower Classification Web App

This is a **Streamlit-based machine learning app** that classifies Iris flowers into three species — *Setosa, Versicolor,* and *Virginica* — using measurements of sepal and petal features.

---

## 🚀 Live App
Click below to open the hosted version on Streamlit Cloud:  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iris-flower-classifier-y35s7iemyzpcuanwyzpci2.streamlit.app/)

---

## 🧾 Project Overview
- **Language:** Python  
- **Libraries:** Streamlit, Pandas, NumPy, Seaborn, Matplotlib, Plotly, Scikit-learn  
- **Algorithms Used:** Random Forest, Logistic Regression, K-Nearest Neighbors  
- **Dataset:** Built-in Iris dataset from Scikit-learn  
- **Goal:** Predict the species of an Iris flower based on user input.

---

## 🛠️ Installation and Setup (Run Locally)

Follow these steps to download the project, install dependencies, and run the Streamlit application on your local machine.

### 🔹 Step 1 — Download the Project

1.  Go to the GitHub repository:
    👉 **https://github.com/kushforges/iris-flower-classifier**
2.  Click the green **“Code”** button and select **“Download ZIP.”**
3.  Once downloaded, **extract** the ZIP file to a folder of your choice (e.g., your Desktop or Downloads folder).

### 🔹 Step 2 — Open the Project in VS Code

1.  Open **Visual Studio Code (VS Code)**.
2.  Click **File → Open Folder...** and select the extracted project folder.
3.  Open the main application file, `iris_app.py`, from the Explorer panel.

### 🔹 Step 3 — Install Required Libraries (Dependencies)

Before running the app, you must install all required Python libraries.

1.  In VS Code, open a new terminal (**Terminal → New Terminal**).
2.  Run the following command:

    ```bash
    pip install -r requirements.txt
    ```

    *(If the `pip` command is not recognized, try: `python -m pip install -r requirements.txt`)*

### 🔹 Step 4 — Run the Streamlit App

1.  In the same terminal, start the Streamlit application:

    ```bash
    python -m streamlit run iris_app.py
    ```

2.  Streamlit will start a local web server and print the URLs. Look for:

    ```
    Local URL: http://localhost:8501
    ```

### 🔹 Step 5 — View the App in Browser

1.  Open your web browser (Chrome, Firefox, etc.).
2.  Visit the local URL: 👉 **http://localhost:8501**
3.  You should now see the interactive Iris Flower Classification web app running locally!

---

## 📂 Repository Contents

| File | Description |
| :--- | :--- |
| `iris_app.py` | Streamlit application code |
| `requirements.txt` | List of dependencies |
| `projectreport.pdf` | Project report |
| `evidence.pdf` | Evidence of Ongoing Product Development / Deployment |
| `README.md` | Project overview and instructions |
