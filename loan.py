import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Loading the models
loaded_model = pickle.load(open('models/loan_classifier.sav', 'rb'))
standard_scaler = pickle.load(open('models/scaler.sav', 'rb'))

def loan_prediction(input_data):
    input_data_as_numpy = np.asarray(input_data)

    input_data_reshape = input_data_as_numpy.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshape)

    if prediction[0]==1:
        return "We're sorry, Your Loan has not been Approved."
    else:
        return "Congratulations, Your Loan has been Approved." 

###Styling
# Streamlit app
def main():
    st.markdown("<h3 style='text-align: center;'>Bank Loan Prediction</h3>", unsafe_allow_html=True)

#Styling my
    st.markdown("<h5 style='color: green; font-size: 16px;'>Input the required values:</h5>", unsafe_allow_html=True)

    # Collecting user input values from end user
    #Creating columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Gender = st.selectbox("Gender (Select 0 or 1)", options=[0, 1])
    
    with col2:
        Married = st.selectbox("Married (Select 0 or 1)", options=[0, 1])
    
    with col3:
        Dependents = st.selectbox("Dependents (0 or 1)", options=[0, 1])
    
    with col1:
        Education = st.selectbox("Education (0 or 1)", options=[0, 1])

    
    with col2:
        Self_Employed = st.selectbox("Employment Status (0 or 1)", options=[0, 1])
        
    
    with col3:
        Applicant_Income = st.number_input('Applicant Income', value=None)
    
    with col1:
        Coapplicant_Income = st.number_input('Coapplicant Income', value=None)
        
    
    with col2:
        Loan_Amount = st.number_input('Loan Amount', value=None)
        
    
    with col3:
        Loan_Amount_Term = st.number_input('Loan Amount Term', value=None)
    
    
    with col1:
        Credit_History = st.number_input('Credit History', value=None)
    
    with col2:
        Property_Area = st.number_input('Property Area', value=None)
        
    
        

    # When user clicks "Predict"
    if st.button('Bank Loan Application'):
        try:
            input_data = [
                int(Gender),
                int(Married),
                int(Dependents),
                int(Education),
                int(Self_Employed),
                int(Applicant_Income),
                float(Coapplicant_Income),
                float(Loan_Amount),
                float(Loan_Amount_Term),
                int(Credit_History),
                int(Property_Area)
            ]
            
            result = loan_prediction(input_data)
            
            st.success(result)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            
            
# Run the app
if __name__ == "__main__":
    main()