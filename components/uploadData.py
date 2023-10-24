import streamlit as st
import pytesseract
from pytesseract import Output
import cv2

def dataTab():
    submitted = False
    buttonState = False
    uploaded_file = None
    provider_first = ""
    provider_last = ""
    submitted_charge = 0
    has_med = ''
    med_payment = 0
#    HCPCS = 0

    provider_firstI = ""
    provider_lastI = ""
    submitted_chargeI = 0
    has_medI = ''
    med_paymentI = 0
 #   HCPCSI = 0
    uploadTab, inputTab = st.tabs(["Upload","Info"])

    st.markdown(
        """
        <style>
            /* Center the text within the button */
            .stButton > button span {
                display: flex;
                align-items: center;
                justify-content: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )  
    with uploadTab:
        uploaded_file = st.file_uploader(
            "Upload a png, pdf, or jpeg file",
            type=["png", "pdf", "jpeg"]
        )
        if uploaded_file != None:
            left, right = st.columns([6,1])
            with right:    
                if st.button("Scan"):
                    buttonState = True
                    provider_firstI, provider_lastI, submitted_chargeI, has_medI, med_paymentI = scanInvoice(uploaded_file)
                    uploaded_file = None
                    
    with inputTab:
        with st.form("Information"):
            if not buttonState:
                st.write("Information")
                provider_first = st.text_input("Provider First Name")
                provider_last = st.text_input("Provider Last Name")
                submitted_charge = st.number_input("Total Cost")
                st.write("Prescribed Medication")
                has_med = st.checkbox('Yes')
                med_payment = st.number_input("Medicare Payment")
                if st.form_submit_button():
                    with open("form_submit_state.txt", "w") as file:
                        file.write("pressed")
            else:
                st.write("Information")
                provider_first = st.text_input("Provider First Name", provider_firstI)
                provider_last = st.text_input("Provider Last Name", provider_lastI)
                submitted_charge = st.number_input("Total Cost", submitted_chargeI)
                st.write("Has Medicare")
                has_med = st.checkbox('Yes')
                med_payment = st.number_input("Medicare Payment", med_paymentI)
                if st.form_submit_button():
                    with open("form_submit_state.txt", "w") as file:
                        file.write("pressed")
    return provider_first, provider_last, submitted_charge, has_med, med_payment 


def scanInvoice(file_name):
    img = cv2.imread(file_name)
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        element = d['text'][i]
    if (element == 'Physician'):
        prov_first = d['text'][i - 8]
        prov_last = d['text'][i - 7]
    if (element == 'PROVIDER'):
        raw_total = d['text'][i + 4]
    if (element == 'MEDICARE'):
        raw_medicare_payment = d['text'][i + 7]
    if (element == 'Patient:'):
        patient_first = d['text'][i + 1]
        patient_last = d['text'][i + 2]
    if float(d['conf'][i]) > 40:  # Check if confidence score is greater than 60
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    medicare_payment = raw_medicare_payment[1:]
    total = raw_total[1:]

    has_medicare = True

    if (medicare_payment == 0):
        has_medicare = False

    return prov_first, prov_last, total, has_medicare, medicare_payment
