import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load pretrained model
model = tf.keras.models.load_model('AbgabeModell.h5')

# Function to predict the tool class
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
                st.success('Klassifizierung abgeschlossen!')
                st.write('Die Werkzeugklasse ist:', tool_class)
                

if __name__ == '__main__':
    main()





