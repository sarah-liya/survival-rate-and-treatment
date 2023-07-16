import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Hide the Streamlit menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu, footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add custom CSS to remove the header space
custom_css = """
            <style>
            body {
                margin-top: 0;
            }
            </style>
            """
st.markdown(custom_css, unsafe_allow_html=True)

st.header('CANCERVIVE: Cancer Survival Rate Prediction System')

st.write('Disclaimer: The Cancer Survival Rate prediction and Treatment Recommendation results is based on Machine Learning Data')

def user_input_features():
    cancer_types = [
        "Bladder",
        "Lung",
        "Breast",
        "Liver",
        "Oesophagus"
    ]
    stages = [
        "1",
        "2",
        "3",
        "4"
    ]
    genders = [
        "Female",
        "Male"
    ]
    ages = [
        "15-44",
        "45-54",
        "55-64",
        "65-74",
        "75-99",
        "All ages"
    ]

    years_options = [
        "1",
        "2",
        "3",
        "4",
        "5"
    ]

    cancer_type = st.selectbox('**Cancer Type**', cancer_types)
    stage = st.selectbox('**Cancer Stage**', stages)
    gender = st.selectbox('**Gender**', genders)
    age = st.selectbox('**Patient Age**', ages)
    selected_years = st.selectbox('**Years Since Diagnosis**', years_options)

    data = {'Cancer type': cancer_type, 'Gender': gender, 'Stage': stage, 'Age At Diagnosis': age,
            'Years Since Diagnosis': selected_years}

    features = pd.DataFrame(data, index=[0])

    return features

# Add the input form
input_features = user_input_features()

# Load the Cancer survival dataset from CSV
url = 'https://raw.githubusercontent.com/sarah-liya/survival-web/main/FYP%20Cancer%20Survival%20new.csv'
cancerS_df = pd.read_csv(url, low_memory=False)

# Encode the categorical features in the original dataset
encode = ['Cancer type', 'Gender', 'Stage', 'Age At Diagnosis']
for col in encode:
    label_encoder = LabelEncoder()
    cancerS_df[col] = label_encoder.fit_transform(cancerS_df[col])
    input_features[col] = label_encoder.transform(input_features[col])

features = ['Cancer type', 'Gender', 'Stage', 'Age At Diagnosis', 'Years Since Diagnosis']
X = cancerS_df[features]
y = cancerS_df['Survival (%)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build Decision Forest Regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

if input_features is not None:

    # Add a button to trigger the prediction
    if st.button('Predict Cancer Survival'):
        # Predict survival
        prediction = rf.predict(input_features)[0]

        # Display the prediction
        st.write('### Patients Survival Rate Prediction')
        st.title(f'{prediction:.2f}%')

        # Extract the attribute weights (feature importances)
        attribute_weights = rf.feature_importances_

        # Create a DataFrame to store the attribute weights
        attribute_weights_df = pd.DataFrame({'Attribute': X_train.columns, 'Weight': attribute_weights})

        # Plot the attribute weights
        plt.figure(figsize=(10, 6))
        plt.bar(attribute_weights_df['Attribute'], attribute_weights_df['Weight'])
        plt.xlabel('Factor')
        plt.ylabel('Factor weight')
        plt.title('Importance factor')
        plt.xticks(rotation=45)
        st.pyplot(plt)

        selected_attributes = ['Stage', 'Gender', 'Age At Diagnosis']

        # Filter the attribute weights based on selected attributes
        filtered_attribute_weights_df = attribute_weights_df[attribute_weights_df['Attribute'].isin(selected_attributes)]

        # Find the attribute with the highest weight
        highest_weight_attribute = filtered_attribute_weights_df.loc[filtered_attribute_weights_df['Weight'].idxmax(), 'Attribute']
        highest_weight = filtered_attribute_weights_df['Weight'].max()
        
        def treatment_recommendation(input_features, highest_weight_attribute):
            # Retrieve the selected cancer type from the input features        
            cancer_type = input_features['Cancer type'].iloc[0]
            
            # Perform treatment recommendation based on conditions
            if highest_weight_attribute == 'Stage':
                if input_features['Cancer type'].iloc[0] == 0:  #Bladder
                    st.write('### Cancer Treatment Recommendation for Bladder Cancer')
                    if input_features['Stage'].iloc[0] == 0:                        
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy')

                        
                elif input_features['Cancer type'].iloc[0] == 1:  #Breast
                    st.write('### Cancer Treatment Recommendation for Breast Cancer')
                    if input_features['Stage'].iloc[0] == 0:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy') 
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- No Treatment Recommend')

                elif input_features['Cancer type'].iloc[0] == 4:  #Oesophagus
                    st.write('### Cancer Treatment Recommendation for Oesophagus Cancer')                    
                    if input_features['Stage'].iloc[0] == 0:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')

                elif input_features['Cancer type'].iloc[0] == 2:  #Liver
                    st.write('### Cancer Treatment Recommendation for Liver Cancer')     
                    if input_features['Stage'].iloc[0] == 0:
                        st.write('- Tumour Resection')
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Chemotherapy')

                elif input_features['Cancer type'].iloc[0] == 3:  #Lung
                    st.write('### Cancer Treatment Recommendation for Lung Cancer')
                    if input_features['Stage'].iloc[0] == 0:             
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Chemotherapy')
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy')
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Chemotherapy')





            elif highest_weight_attribute == 'Gender':
                if input_features['Cancer type'].iloc[0] == 0:  #Bladder
                    st.write('### Cancer Treatment Recommendation for Bladder Cancer')
                    if input_features['Stage'].iloc[0] == 0:                        
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy')

                        
                elif input_features['Cancer type'].iloc[0] == 1:  #Breast
                    st.write('### Cancer Treatment Recommendation for Breast Cancer')
                    if input_features['Stage'].iloc[0] == 0:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy') 
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- No Treatment Recommend')

                elif input_features['Cancer type'].iloc[0] == 4:  #Oesophagus
                    st.write('### Cancer Treatment Recommendation for Oesophagus Cancer')                    
                    if input_features['Stage'].iloc[0] == 0:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')

                elif input_features['Cancer type'].iloc[0] == 2:  #Liver
                    st.write('### Cancer Treatment Recommendation for Liver Cancer')     
                    if input_features['Stage'].iloc[0] == 0:
                        st.write('- Tumour Resection')
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Chemotherapy')

                elif input_features['Cancer type'].iloc[0] == 3:  #Lung
                    st.write('### Cancer Treatment Recommendation for Lung Cancer')
                    if input_features['Stage'].iloc[0] == 0:             
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy')
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy')
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Chemotherapy')
            elif highest_weight_attribute == 'Age At Diagnosis':
                if input_features['Cancer type'].iloc[0] == 0:  #Bladder
                    st.write('### Cancer Treatment Recommendation for Bladder Cancer')
                    if input_features['Stage'].iloc[0] == 0:                        
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy')

                        
                elif input_features['Cancer type'].iloc[0] == 1:  #Breast
                    st.write('### Cancer Treatment Recommendation for Breast Cancer')
                    if input_features['Stage'].iloc[0] == 0:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy') 
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- No Treatment Recommend')

                elif input_features['Cancer type'].iloc[0] == 4:  #Oesophagus
                    st.write('### Cancer Treatment Recommendation for Oesophagus Cancer')                    
                    if input_features['Stage'].iloc[0] == 0:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Tumour Resection')
                        st.write('- Radiotherapy')

                elif input_features['Cancer type'].iloc[0] == 2:  #Liver
                    st.write('### Cancer Treatment Recommendation for Liver Cancer')     
                    if input_features['Stage'].iloc[0] == 0:
                        st.write('- Tumour Resection')
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Tumour Resection')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Chemotherapy')

                elif input_features['Cancer type'].iloc[0] == 3:  #Lung
                    st.write('### Cancer Treatment Recommendation for Lung Cancer')
                    if input_features['Stage'].iloc[0] == 0:             
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy') 
                    elif input_features['Stage'].iloc[0] == 1:
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy')
                    elif input_features['Stage'].iloc[0] == 2:
                        st.write('- Radiotherapy')
                        st.write('- Chemotherapy')
                    elif input_features['Stage'].iloc[0] == 3:
                        st.write('- Chemotherapy')
        # Call treatment_recommendation function
        treatment_recommendation(input_features, highest_weight_attribute)
