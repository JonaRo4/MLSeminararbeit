import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd

# Load pretrained model
model = tf.keras.models.load_model('AbgabeModell.h5')

# Function to preprocess the image and predict the tool class
def predict_tool_class(image):
    try:
        image = image.resize((224, 224))  # Resize image
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image)
        classes = ['Defekt', 'Mittel', 'Neu']
        predicted_class = classes[np.argmax(prediction)]
        return predicted_class
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Function to load data from Excel file
def load_data_from_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the Excel file: {e}")
        return None

# Streamlit App
def main():
    st.title('Werkzeugklassifizierung')

    uploaded_image = st.file_uploader("Bild von Werkzeug hochladen", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Hochgeladenes Bild', use_column_width=True)

        tool_name = st.text_input("Werkzeugname", "")
        processing_time = st.number_input("Bearbeitungsdauer (in Minuten)", min_value=0)
        material = st.text_input("Material des Werkzeugs", "")

        if st.button('Klassifizieren'):
            with st.spinner('Klassifizierung läuft...'):
                tool_class = predict_tool_class(image)
                if tool_class:
                    st.success('Klassifizierung abgeschlossen!')
                    st.write('Die Werkzeugklasse ist:', tool_class)
                    
                    # Saving the data to a DataFrame
                    data = {'Werkzeugname': [tool_name],
                            'Bearbeitungsdauer (Minuten)': [processing_time],
                            'Material des Werkzeugs': [material],
                            'Werkzeugklasse': [tool_class]}
                    df = pd.DataFrame(data)

                    # Load data from Excel file
                    excel_file_path = "Exceltest.xlsx"
                    excel_data = load_data_from_excel(excel_file_path)

                    if excel_data is not None:
                        # Append new data to the Excel data
                        updated_data = pd.concat([excel_data, df], ignore_index=True)
                        # Display the updated DataFrame
                        st.write('Aktualisierte Daten aus der Excel-Datei:')
                        st.dataframe(updated_data)
                else:
                    st.error('Fehler bei der Klassifizierung. Bitte versuche es erneut.')

if __name__ == '__main__':
    main()

