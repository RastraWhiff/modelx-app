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
Applicant_Age = st.selectbox("Applicant Age", [
    "21 Years", "22 Years", "23 Years", "24 Years", "25 Years", "26 Years", "27 Years", 
    "28 Years", "29 Years", "30 Years", "31 Years", "32 Years", "33 Years", "34 Years", 
    "35 Years", "36 Years", "37 Years", "38 Years", "39 Years", "40 Years", "41 Years", 
    "42 Years", "43 Years", "44 Years", "45 Years", "46 Years", "47 Years", "48 Years", 
    "49 Years", "50 Years", "51 Years", "52 Years", "53 Years", "54 Years", "55 Years", 
    "56 Years", "57 Years", "58 Years", "59 Years", "60 Years", "61 Years", "62 Years", 
    "63 Years", "64 Years", "65 Years", "66 Years", "67 Years", "68 Years", "69 Years", 
    "70 Years", "71 Years", "72 Years", "73 Years", "74 Years", "75 Years", "76 Years", 
    "77 Years", "78 Years", "79 Years"
])
Work_Experience = st.selectbox("Work Experience", [
    "0 Years", "1 Year", "2 Years", "3 Years", "4 Years", "5 Years", "6 Years", "7 Years", 
    "8 Years", "9 Years", "10 Years", "11 Years", "12 Years", "13 Years", "14 Years", 
    "15 Years", "16 Years", "17 Years", "18 Years", "19 Years", "20 Years"
])
Marital_Status = st.radio("Marital Status", ["Married", "Single"])  # Changed to descriptive options
House_Ownership = st.selectbox("House Ownership", ["Not Both", "Owned", "Rented"])  # Changed to new options
Vehicle_Ownership_Car = st.selectbox("Vehicle Ownership (Car)", ["No", "Yes"])  # Changed to descriptive options
Occupation = st.selectbox("Occupation", [
    "Air traffic controller", "Analyst", "Architect", "Army officer", "Artist", "Aviator", 
    "Biomedical Engineer", "Chartered Accountant", "Chef", "Chemical engineer", 
    "Civil engineer", "Civil servant", "Comedian", "Computer hardware engineer", 
    "Computer operator", "Consultant", "Dentist", "Design Engineer", "Designer", 
    "Drafter", "Economist", "Engineer", "Fashion Designer", "Financial Analyst", 
    "Firefighter", "Flight attendant", "Geologist", "Graphic Designer", "Hotel Manager", 
    "Industrial Engineer", "Lawyer", "Librarian", "Magistrate", "Mechanical engineer", 
    "Microbiologist", "Official", "Petroleum Engineer", "Physician", "Police officer", 
    "Politician", "Psychologist", "Scientist", "Secretary", "Software Developer", 
    "Statistician", "Surgeon", "Surveyor", "Technical writer", "Technician", 
    "Technology specialist", "Web designer"
])
Years_in_Current_Employment = st.selectbox("Years in Current Employment", [
    "0 Years", "1 Year", "2 Years", "3 Years", "4 Years", "5 Years", "6 Years", "7 Years", 
    "8 Years", "9 Years", "10 Years", "11 Years", "12 Years", "13 Years", "14 Years"
])
Years_in_Current_Residence = st.selectbox("Years in Current Residence", [
    "10 Years", "11 Years", "12 Years", "13 Years", "14 Years"
])
Annual_Income_IDR = st.number_input("Annual Income (IDR)", min_value=0, value=50000000)


# Map categorical inputs to numeric values
Applicant_Age_mapping = {
    "21 Years": 0, "22 Years": 1, "23 Years": 2, "24 Years": 3, "25 Years": 4, "26 Years": 5, 
    "27 Years": 6, "28 Years": 7, "29 Years": 8, "30 Years": 9, "31 Years": 10, "32 Years": 11, 
    "33 Years": 12, "34 Years": 13, "35 Years": 14, "36 Years": 15, "37 Years": 16, "38 Years": 17, 
    "39 Years": 18, "40 Years": 19, "41 Years": 20, "42 Years": 21, "43 Years": 22, "44 Years": 23, 
    "45 Years": 24, "46 Years": 25, "47 Years": 26, "48 Years": 27, "49 Years": 28, "50 Years": 29, 
    "51 Years": 30, "52 Years": 31, "53 Years": 32, "54 Years": 33, "55 Years": 34, "56 Years": 35, 
    "57 Years": 36, "58 Years": 37, "59 Years": 38, "60 Years": 39, "61 Years": 40, "62 Years": 41, 
    "63 Years": 42, "64 Years": 43, "65 Years": 44, "66 Years": 45, "67 Years": 46, "68 Years": 47, 
    "69 Years": 48, "70 Years": 49, "71 Years": 50, "72 Years": 51, "73 Years": 52, "74 Years": 53, 
    "75 Years": 54, "76 Years": 55, "77 Years": 56, "78 Years": 57, "79 Years": 58
}
Work_Experience_mapping = {
    "0 Years": 0, "1 Year": 1, "2 Years": 2, "3 Years": 3, "4 Years": 4, "5 Years": 5, 
    "6 Years": 6, "7 Years": 7, "8 Years": 8, "9 Years": 9, "10 Years": 10, "11 Years": 11, 
    "12 Years": 12, "13 Years": 13, "14 Years": 14, "15 Years": 15, "16 Years": 16, "17 Years": 17, 
    "18 Years": 18, "19 Years": 19, "20 Years": 20
}
marital_status_mapping = {"Married": 0, "Single": 1}
house_ownership_mapping = {"Not Both": 0, "Owned": 1, "Rented": 2}
vehicle_ownership_car_mapping = {"No": 0, "Yes": 1}
occupation_mapping = {
    "Air traffic controller": 0, "Analyst": 1, "Architect": 2, "Army officer": 3, "Artist": 4, "Aviator": 5, 
    "Biomedical Engineer": 6, "Chartered Accountant": 7, "Chef": 8, "Chemical engineer": 9, 
    "Civil engineer": 10, "Civil servant": 11, "Comedian": 12, "Computer hardware engineer": 13, 
    "Computer operator": 14, "Consultant": 15, "Dentist": 16, "Design Engineer": 17, "Designer": 18, 
    "Drafter": 19, "Economist": 20, "Engineer": 21, "Fashion Designer": 22, "Financial Analyst": 23, 
    "Firefighter": 24, "Flight attendant": 25, "Geologist": 26, "Graphic Designer": 27, "Hotel Manager": 28, 
    "Industrial Engineer": 29, "Lawyer": 30, "Librarian": 31, "Magistrate": 32, "Mechanical engineer": 33, 
    "Microbiologist": 34, "Official": 35, "Petroleum Engineer": 36, "Physician": 37, "Police officer": 38, 
    "Politician": 39, "Psychologist": 40, "Scientist": 41, "Secretary": 42, "Software Developer": 43, 
    "Statistician": 44, "Surgeon": 45, "Surveyor": 46, "Technical writer": 47, "Technician": 48, 
    "Technology specialist": 49, "Web designer": 50
}
Years_in_Current_Employment_mapping = {
    "0 Years": 0, "1 Year": 1, "2 Years": 2, "3 Years": 3, "4 Years": 4, "5 Years": 5, 
    "6 Years": 6, "7 Years": 7, "8 Years": 8, "9 Years": 9, "10 Years": 10, "11 Years": 11, 
    "12 Years": 12, "13 Years": 13, "14 Years": 14
}
Years_in_Current_Residence_mapping = {
    "10 Years": 0, "11 Years": 1, "12 Years": 2, "13 Years": 3, "14 Years": 4
}

# Prediction button
if st.button("Predict"):
    new_data = np.array([[
        Applicant_Age, 
        Work_Experience, 
        marital_status_mapping[Marital_Status], 
        house_ownership_mapping[House_Ownership], 
        vehicle_ownership_car_mapping[Vehicle_Ownership_Car], 
        occupation_mapping[Occupation], 
        Years_in_Current_Employment, 
        Years_in_Current_Residence, 
        Annual_Income_IDR
    ]])
    
    result = xgb_model.predict(new_data)
    result_label = LABEL[int(result[0])]

    st.write("Prediction Result: ", result_label)











# BENAR
# import streamlit as st
# import numpy as np
# import xgboost as xgb
# import pickle

# # Load the model
# with open("xgb_model.pkl", "rb") as model_file:
#     xgb_model = pickle.load(model_file)

# LABEL = ['Bisa Meminjam (0)', "Tidak Bisa Meminjam (1)"]

# # Streamlit app
# st.title("Loan Eligibility Prediction")

# # Create input fields
# Applicant_Age = st.number_input("Applicant Age", min_value=0, max_value=100, value=30)
# Work_Experience = st.number_input("Work Experience", min_value=0, max_value=50, value=5)
# Marital_Status = st.selectbox("Marital Status", [0, 1])  # 0: Single, 1: Married (you can customize as needed)
# House_Ownership = st.selectbox("House Ownership", [0, 1])  # 0: No, 1: Yes (you can customize as needed)
# Vehicle_Ownership_Car = st.selectbox("Vehicle Ownership (Car)", [0, 1])  # 0: No, 1: Yes (you can customize as needed)
# Occupation = st.number_input("Occupation", min_value=0, max_value=10, value=1)  # Customize as needed
# Years_in_Current_Employment = st.number_input("Years in Current Employment", min_value=0, max_value=50, value=10)
# Years_in_Current_Residence = st.number_input("Years in Current Residence", min_value=0, max_value=50, value=5)
# Annual_Income_IDR = st.number_input("Annual Income (IDR)", min_value=0, value=50000000)

# # Prediction button
# if st.button("Predict"):
#     new_data = np.array([[Applicant_Age, Work_Experience, Marital_Status, House_Ownership, Vehicle_Ownership_Car, Occupation, Years_in_Current_Employment, Years_in_Current_Residence, Annual_Income_IDR]])
#     result = xgb_model.predict(new_data)
#     result_label = LABEL[int(result[0])]

#     st.write("Prediction Result: ", result_label)




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
