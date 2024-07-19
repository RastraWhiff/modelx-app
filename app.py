import streamlit as st
import numpy as np
import xgboost as xgb
import pickle

# Load the model
with open("xgb_model.pkl", "rb") as model_file:
    xgb_model = pickle.load(model_file)

LABEL = ['Bisa Meminjam (0)', "Tidak Bisa Meminjam (1)"]

# Streamlit app
st.title("Loan Eligibility Prediction")

# Create input fields
Applicant_Age = st.number_input("Applicant Age", min_value=0, max_value=100, value=30)
Work_Experience = st.number_input("Work Experience", min_value=0, max_value=50, value=5)
Marital_Status = st.selectbox("Marital Status", [0, 1])  # 0: Single, 1: Married (you can customize as needed)
House_Ownership = st.selectbox("House Ownership", [0, 1])  # 0: No, 1: Yes (you can customize as needed)
Vehicle_Ownership_Car = st.selectbox("Vehicle Ownership (Car)", [0, 1])  # 0: No, 1: Yes (you can customize as needed)
Occupation = st.number_input("Occupation", min_value=0, max_value=10, value=1)  # Customize as needed
Years_in_Current_Employment = st.number_input("Years in Current Employment", min_value=0, max_value=50, value=10)
Years_in_Current_Residence = st.number_input("Years in Current Residence", min_value=0, max_value=50, value=5)
Annual_Income_IDR = st.number_input("Annual Income (IDR)", min_value=0, value=50000000)

# Prediction button
if st.button("Predict"):
    new_data = np.array([[Applicant_Age, Work_Experience, Marital_Status, House_Ownership, Vehicle_Ownership_Car, Occupation, Years_in_Current_Employment, Years_in_Current_Residence, Annual_Income_IDR]])
    result = xgb_model.predict(new_data)
    result_label = LABEL[int(result[0])]

    st.write("Prediction Result: ", result_label)




# import numpy as np
# import xgboost as xgb
# import pandas as pd  # tambahkan jika menggunakan pandas untuk pemrosesan data
# from flask import Flask, render_template, request

# app = Flask(__name__)

# # load model
# xgb_model = xgb.Booster()
# xgb_model.load_model("model/xgb_model.json")  # ubah sesuai dengan lokasi dan nama model yang sebenarnya

# LABEL = ['Bisa Meminjam (0)', "Tidak Bisa Meminjam (1)"]

# @app.route("/")
# def data():
#     return render_template("index.html")
     
# @app.route("/predict", methods=['POST'])   
# def predict():
#     # getting input with name in HTML form
#     Applicant_Age = float(request.form.get("Applicant_Age"))
#     Work_Experience = float(request.form.get("Work_Experience"))
#     Marital_Status = float(request.form.get("Marital_Status"))
#     House_Ownership = float(request.form.get("House_Ownership")) 
#     Vehicle_Ownership_Car = float(request.form.get("Vehicle_Ownership_Car"))
#     Occupation = float(request.form.get("Occupation"))
#     Years_in_Current_Employment = float(request.form.get("Years_in_Current_Employment"))
#     Years_in_Current_Residence = float(request.form.get("Years_in_Current_Residence"))
#     Annual_Income_IDR = float(request.form.get("Annual_Income_IDR"))

#     # Membuat DataFrame baru dengan data input
#     new_data = pd.DataFrame([[Applicant_Age, Work_Experience, Marital_Status, House_Ownership, 
#                               Vehicle_Ownership_Car, Occupation, Years_in_Current_Employment, 
#                               Years_in_Current_Residence, Annual_Income_IDR]],
#                             columns=['Applicant_Age', 'Work_Experience', 'Marital_Status', 
#                                      'House_Ownership', 'Vehicle_Ownership_Car', 'Occupation', 
#                                      'Years_in_Current_Employment', 'Years_in_Current_Residence', 
#                                      'Annual_Income_IDR'])

#     # Mengubah DataFrame menjadi DMatrix
#     dnew = xgb.DMatrix(new_data)

#     # Melakukan prediksi
#     prediction = xgb_model.predict(dnew)
#     result = LABEL[1] if prediction > 0.5 else LABEL[0]

#     return render_template("index.html", prediction_result=result, 
#                            Applicant_Age=Applicant_Age, Work_Experience=Work_Experience,
#                            Marital_Status=Marital_Status, House_Ownership=House_Ownership,
#                            Vehicle_Ownership_Car=Vehicle_Ownership_Car, Occupation=Occupation,
#                            Years_in_Current_Employment=Years_in_Current_Employment,
#                            Years_in_Current_Residence=Years_in_Current_Residence,
#                            Annual_Income_IDR=Annual_Income_IDR)

# if __name__ == "__main__":
#     app.run(debug=True)




# import numpy as np
# import xgboost as xgb
# import pickle
# from flask import Flask
# from flask import render_template
# from flask import request

# app = Flask(__name__)

# # Muat model menggunakan load_model
# loaded_model = xgb.XGBClassifier()
# loaded_model.load_model("model/xgb_model.json")
# LABEL = ['Bisa Meminjam (0)', "Tidak Bisa Meminjam (1)"]


# @app.route("/")
# def data():
#         return render_template("index.html")
     
# @app.route("/predict", methods=['POST'])   
# def predict():
#     # getting input with name in HTML form dan ubah dalam bentuk float 
#        Applicant_Age = float(request.form.get("Applicant_Age"))
#        Work_Experience = float(request.form.get("Work_Experience"))
#        Marital_Status = float(request.form.get("Marital_Status"))
#        House_Ownership = float(request.form.get("House_Ownership")) 
#        Vehicle_Ownership_Car = float(request.form.get("Vehicle_Ownership_Car"))
#        Occupation = float(request.form.get("Occupation"))
#        Years_in_Current_Employment = float(request.form.get("Years_in_Current_Employment"))
#        Years_in_Current_Residence = float(request.form.get("Years_in_Current_Residence"))
#        Annual_Income_IDR = float(request.form.get("Annual_Income_IDR"))
#        # Print the text in terminal for verification 
#         # print(sepal_length)
#        # Membentuk data baru
#        new_data = [[Applicant_Age, Work_Experience, Marital_Status, House_Ownership, Vehicle_Ownership_Car, Occupation, Years_in_Current_Employment, Years_in_Current_Residence, Annual_Income_IDR]]
    
#        # Prediksi menggunakan model
#        prediction = loaded_model.predict(new_data)
#        result = LABEL[int(prediction[0])]
    
#        return render_template("index.html", prediction_result=result, Applicant_Age=Applicant_Age,Work_Experience=Work_Experience,Marital_Status=Marital_Status,House_Ownership=House_Ownership,Vehicle_Ownership_Car=Vehicle_Ownership_Car,Occupation=Occupation,Years_in_Current_Employment=Years_in_Current_Employment,Years_in_Current_Residence=Years_in_Current_Residence,Annual_Income_IDR=Annual_Income_IDR) 
       
# if __name__ == "__main__":
#     app.run(debug=True)

# import numpy as np
# import xgboost as xgb
# import pickle
# from flask import Flask, render_template, request

# app = Flask(__name__)

# # Muat model XGBoost dari file JSON
# loaded_model = xgb.Booster()
# loaded_model.load_model("model/xgb_model.json")

# LABEL = ['Bisa Meminjam (0)', 'Tidak Bisa Meminjam (1)']

# @app.route("/")
# def data():
#     return render_template("index.html")

# @app.route("/predict", methods=['POST'])
# def predict():
#     # Mendapatkan input dari form HTML dan mengubahnya menjadi float
#     Applicant_Age = float(request.form.get("Applicant_Age"))
#     Work_Experience = float(request.form.get("Work_Experience"))
#     Marital_Status = float(request.form.get("Marital_Status"))
#     House_Ownership = float(request.form.get("House_Ownership"))
#     Vehicle_Ownership_Car = float(request.form.get("Vehicle_Ownership_Car"))
#     Occupation = float(request.form.get("Occupation"))
#     Years_in_Current_Employment = float(request.form.get("Years_in_Current_Employment"))
#     Years_in_Current_Residence = float(request.form.get("Years_in_Current_Residence"))
#     Annual_Income_IDR = float(request.form.get("Annual_Income_IDR"))

#     # Membentuk data baru dalam bentuk numpy array
#     new_data = np.array([[Applicant_Age, Work_Experience, Marital_Status, House_Ownership,
#                           Vehicle_Ownership_Car, Occupation, Years_in_Current_Employment,
#                           Years_in_Current_Residence, Annual_Income_IDR]])

#     # Konversi data ke DMatrix
#     new_data_dmatrix = xgb.DMatrix(new_data)

#     # Prediksi menggunakan model
#     prediction = loaded_model.predict(new_data_dmatrix)
#     result = LABEL[int(prediction[0])]

#     # Render template dengan hasil prediksi dan input data
#     return render_template("index.html", prediction_result=result,
#                            Applicant_Age=Applicant_Age, Work_Experience=Work_Experience,
#                            Marital_Status=Marital_Status, House_Ownership=House_Ownership,
#                            Vehicle_Ownership_Car=Vehicle_Ownership_Car, Occupation=Occupation,
#                            Years_in_Current_Employment=Years_in_Current_Employment,
#                            Years_in_Current_Residence=Years_in_Current_Residence,
#                            Annual_Income_IDR=Annual_Income_IDR)

# if __name__ == "__main__":
#     app.run(debug=True)
