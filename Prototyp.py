import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Laden des trainierten Modells
model = load_model("FM.h5")

# Klassenlabels definieren
class_labels = ['Neu', 'Mittel', 'Defekt']

# Streamlit-App definieren
def main():
    st.title('Werkzeugklassifizierung')
    
    # Bild hochladen
    uploaded_image = st.file_uploader("Bild von Werkzeug hochladen", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Hochgeladenes Bild', use_column_width=True)
        
        # Vorschub, Drehzahl, Zustellung und Werkzeugtyp eingeben
        vorschub = st.number_input('Vorschub in mm/min', min_value=0, step=1)
        drehzahl = st.number_input('Drehzahl in min-1', min_value=0, step=1)
        zustellung = st.text_input('Zustellung in mm', value='0.0')
        werkzeugtyp = st.text_input('Werkzeugtyp eingeben', '')
        
        try:
            zustellung = float(zustellung)
            assert zustellung >= 0  # Überprüfen, dass die Eingabe nicht negativ ist
        except (ValueError, AssertionError):
            st.error('Bitte geben Sie eine gültige positive Zahl für die Zustellung ein.')
            return
        
        # Nutzungszeit des Werkzeugs eingeben
        nutzungszeit = st.number_input('Nutzungszeit des Werkzeugs in Stunden', min_value=0.0, step=0.5)
        
        # Klassifizierung des hochgeladenen Bildes
        if st.button('Klassifizieren'):
            image = Image.open(uploaded_image)
            image = image.resize((224, 224))  # Größe an das Modell anpassen
            
            # Vorverarbeitung des Bildes
            image = np.array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            
            # Vorhersage machen
            prediction = model.predict(image)
            predicted_class = class_labels[np.argmax(prediction)]
            
            # Ergebnisse anzeigen
            st.write('Werkzeugzustand:', predicted_class)

# Streamlit-App starten
if __name__ == '__main__':
    main()




