# **💉 Diabetes Prediction Using Machine Learning**

A complete Machine Learning project to predict whether a person is diabetic using medical and health-based features such as glucose level, BMI, age, insulin, blood pressure, and more.  
This project demonstrates data preprocessing, model building, evaluation, and prediction using Python and Scikit-Learn.

---

## 🧪 **Project Overview**
This project uses the PIMA Diabetes Dataset to build a classification model that predicts diabetes.  
The notebook includes:
- Data Cleaning  
- Feature Scaling  
- Model Training  
- Model Evaluation  
- Making New Predictions  

It is designed for students, beginners, and anyone learning Machine Learning.

---

## 📂 **Repository Structure**

Diabetes-Prediction/ │ ├── Diabetes_Prediction.ipynb   ← Full ML pipeline notebook
├── .gitignore                  ← Ignore unnecessary files
├── README.md                   ← Project overview
└── requirements.txt            ← Dependencies (optional)

---

## 🔧 **How to Run the Project**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ashishbiswal8658-star/Diabetes-Prediction.git
cd Diabetes-Prediction

2️⃣ Create Virtual Environment (Optional)

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

3️⃣ Install Required Libraries

pip install -r requirements.txt

4️⃣ Run the Notebook

Open the Diabetes_Prediction.ipynb file in Jupyter Notebook or JupyterLab and execute all cells.

```
📝 What the Notebook Contains

Load and Explore Dataset 📊

Handle Missing Values

Standardize Data using StandardScaler

Train ML Model (Logistic Regression / Random Forest / etc.)

Evaluate Model (Accuracy, Precision, Recall)

Predict using New Input Data



---

🚀 Sample Prediction Code

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # or your selected model

# Example input values
input_data = (3, 126, 88, 41, 235, 39.3, 0.704, 27)
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
           'Insulin','BMI','DiabetesPedigreeFunction','Age']

input_df = pd.DataFrame([input_data], columns=columns)
std_data = scaler.transform(input_df)
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")


---

📈 Model Performance

(Replace with your actual output after training)
Example:

Accuracy: 0.78

Precision: 0.75

Recall: 0.72



---

👨‍💻 Author

Ashish Biswal
🔗 GitHub: https://github.com/ashishbiswal8658-star
⭐ Feel free to star this repository and contribute!


---

📄 License

This project is licensed under the MIT License.
