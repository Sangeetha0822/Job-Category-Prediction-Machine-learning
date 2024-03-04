import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Load the data
df = pd.read_csv("C:\\Users\\DELL\\Desktop\\ML mini application\\job_data_merged_1.csv")
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("donloads.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

df.drop("Unnamed: 0", axis=1, inplace=True)
df['Country'] = df['Location'].str.split(',').str[-1].str.strip()

# Model training
X = df[['Workplace', 'Type', 'Department']]
y = df['Category']

X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature names after one-hot encoding
feature_names = X_encoded.columns.tolist()

def predict_category(workplace, job_type, department):
    input_data = pd.DataFrame({'Workplace': [workplace], 'Type': [job_type], 'Department': [department]})
    input_encoded = pd.get_dummies(input_data)
    
    # Ensure all columns from training data are present in input data
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_names]  # Reorder columns to match the training data
    prediction = model.predict(input_encoded)
    return prediction[0]

# Streamlit UI
st.title("Job Category Prediction")

workplace = st.selectbox("Workplace:", ['Select One','Remote', 'Hybrid', 'Onsite'])

# User selection for Type
job_type = st.selectbox(" Type:",['Select One','Temporary','FullTime','Partime','Contract'] )

# Unique departments
department = st.text_input(" Department:")

if st.button("Predict"):
    prediction = predict_category(workplace, job_type, department)
    st.success(f"The predicted job category is: {prediction}")
