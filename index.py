import streamlit as st
#import pickle

import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import joblib
import tensorflow as tf

#for data preprocessing
from sklearn.decomposition import PCA

#for modeling
#from sklearn.neighbors import LocalOutlierFactor
#from sklearn.ensemble import IsolationForest

#filter warnings
import warnings

from components.sidebar import sidebar
from components.uploadData import dataTab
from components.medical_bill_fraud_detection import preprocessing, removeComma

#from joblib import load

from bardapi import Bard
import os

#with open('components/classifier.pkl', 'rb') as file:
#    ae = pickle.load(file)

#ae = joblib.load("components/classifier.pkl")
#ae = ML()
ae = tf.keras.models.load_model("components/path_to_saved_model")


#y_scores = pd.Series(ae.decision_scores_)
risk = 0
st.set_page_config(page_title="MediGuard", page_icon="📖", layout="wide")
st.markdown("<h1 style='text-align: center;'>📖 MediGuard</h1>", unsafe_allow_html=True)

provider_first, provider_last, submitted_charge, has_med, med_payment = dataTab()
sidebar()

with open("form_submit_state.txt", "r") as file:
    form_submit_state = file.read()

def runModel(ae, pF, pL, sC, hM, mP):
    raw_data = pd.read_csv("components/Healthcare Providers.csv")
    data_processed = pd.read_csv("components/Healthcare Providers.csv")
    data_processed = preprocessing(data_processed)
    reconstructions = ae.predict(tf.constant(data_processed))
    
    train_loss = tf.keras.losses.mae(reconstructions, data_processed)
    threshold = np.mean(train_loss) + (np.std(train_loss))

    # user input

    provider_first = str(pF)
    provider_last = str(pL)
    submitted_charge = float(sC)
    if hM:
        med_payment = float(mP)
    else:
        med_payment = 0

    # find the row data of the provider
    row = raw_data.loc[(raw_data['Last Name/Organization Name of the Provider'] == provider_last)
                & (raw_data['First Name of the Provider'] == provider_first)].copy()

    # find the row index of the provider
    row_index = raw_data.index[(raw_data['Last Name/Organization Name of the Provider'] == provider_last)
                & (raw_data['First Name of the Provider'] == provider_first)][0]

    # save the old score before finding the new one
    old_score = train_loss[None, row_index]
    old_score = old_score.numpy()[0]

    # update the row
    #row['Number of Services'] = str(int(row['Number of Services']) + 1)
    row['Number of Services'].iloc[0] = str(int(row['Number of Services'].iloc[0]) + 1)

    if (hM == 1):
        row['Average Medicare Payment Amount'].iloc[0] = str((float(row['Average Medicare Payment Amount'].iloc[0]) * int(row['Number of Medicare Beneficiaries'].iloc[0]) + med_payment) / (int(row['Number of Medicare Beneficiaries'].iloc[0]) + 1))
        row['Number of Medicare Beneficiaries'] = str(int(row['Number of Medicare Beneficiaries']) + 1)
    row['Average Submitted Charge Amount'].iloc[0] = str((float(row['Average Submitted Charge Amount'].iloc[0]) * (int(row['Number of Services'].iloc[0]) - 1) + submitted_charge) / int(row['Number of Services'].iloc[0]))

    temp_data = pd.concat([row,raw_data], ignore_index=True)

    temp_data = preprocessing(temp_data)

    #temp_data.head()

    temp_row = temp_data.iloc[0,:]
    temp_row = temp_row.to_numpy()
    temp_row = temp_row[np.newaxis, :]

    train_loss = tf.keras.losses.mae(ae.predict(temp_row), temp_row)

    new_score = train_loss.numpy()[0]
    new_score = new_score - 0.175

    result = ae.predict(temp_row)
    print(new_score)
    print(old_score)
    print(result)

    if (new_score > old_score and new_score >= threshold):
        return 2 #High Risk
    elif (new_score > old_score and new_score < threshold or new_score < old_score and new_score >= threshold):
        return 1
    else:
        return 0 #Low Risk

def email():
    # Input for recipient's name
    os.environ['_BARD_API_KEY']="cQjrrvI4IsS4yI2ej6Chl3OJSyaa-0EQoQpmApIJPhivIW2EgEQEeBUsFUrv1dDZtpxiEw."
    recipient_name = input("Recipient's Name: ")

    # Input for hospital/clinic name
    hospital_name = input("Hospital/Clinic Name: ")

    # Input for date of service
    date_of_service = input("Date of Service: ")

    # Input for first error details
    first_error = input("Description of the first error: ")

    # Input for second error details (optional)
    second_error = input("Description of the second error (optional): ")

    # Input for additional errors (if applicable, optional)
    additional_errors = input("Description of additional errors (if applicable, optional): ")

    # Input for your name
    your_name = input("Your Name: ")

    # Input for your address
    your_address = input("Your Address: ")

    # Input for city, state, ZIP code
    city_state_zip = input("City, State, ZIP Code: ")

    # Input for your phone number
    your_phone_number = input("Your Phone Number: ")

    # Input for your email address
    your_email_address = input("Your Email Address: ")


    prompt = "Can you draft an email to my doctor " + recipient_name + " from " + hospital_name + "about Medical Billing Errors, specifically that " + first_error + " " + second_error + " " + additional_errors + ". Be sure to include at the end my details: my name: " + your_name + ", my address: " + your_address + ", my city, state, and zipcode: " + city_state_zip + ", my phone number: " + your_phone_number + ", and my email address: " + your_email_address
    prompt = prompt.replace("\r\n", "")

    # Get the answer from Bard
    answer = Bard().get_answer(prompt)['content']


    print(answer)

if form_submit_state == "pressed":
    with open("form_submit_state.txt", "w") as file:
        pass
    risk = runModel(ae, provider_first, provider_last, submitted_charge, has_med, med_payment)
    if (risk == 0):
        st.warning(
            "Low risk of fraud or error detected in the medical bill :)"
        )
        
        print("Low risk of fraud or error detected in the medical bill :)")
    if (risk == 1): 
        st.warning(
            "Potential risk of fraud or error detected in the medical bill."
        )
        email()
        print("Potential risk of fraud or error detected in the medical bill.")
    if (risk == 2):
        st.warning(
            "High risk of fraud or error detected in the medical bill."
        )
        print("High risk of fraud or error detected in the medical bill.")
        email()
    




