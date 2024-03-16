import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Laden des vortrainierten Modells
model = tf.keras.models.load_model('AbgabeModell.h5')

# Funktion zur Vorhersage der Werkzeugklasse
def predict_tool_class(image):
    # Vorverarbeitung des Bildes
    image = image.resize((224, 224))  # Anpassen der Bildgröße
    image = np.array(image) / 255.0  # Normalisierung
    image = np.expand_dims(image, axis=0)  # Hinzufügen einer Dimension für den Batch

    # Vorhersage mit dem Modell
    prediction = model.predict(image)

    # Klassenbezeichnungen
    classes = ['Defekt', 'Mittel', 'Neu']
    predicted_class = classes[np.argmax(prediction)]

    return predicted_class

# Streamlit-App
def main():
    st.title('Werkzeugklassifizierung')

    # Bild hochladen
    uploaded_image = st.file_uploader("Bild von Werkzeug hochladen", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Bild anzeigen
        image = Image.open(uploaded_image)
        st.image(image, caption='Hochgeladenes Bild', use_column_width=True)

        # Zusätzliche Parameter eingeben
        tool_name = st.text_input("Werkzeugname", "")
        processing_time = st.number_input("Bearbeitungsdauer (in Minuten)", min_value=0)
        material = st.text_input("Werkstoff", "")

        # Klassifizierung vornehmen, wenn der Benutzer auf den Button klickt
        if st.button('Klassifizieren'):
            tool_class = predict_tool_class(image)
            st.write('Die Werkzeugklasse ist:', tool_class)
            st.write('Werkzeugname:', tool_name)
            st.write('Bearbeitungsdauer:', processing_time, 'Minuten')
            st.write('Werkstoff:', material)

if __name__ == '__main__':
    main()

