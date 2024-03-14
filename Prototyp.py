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
        zustellung = st.number_input('Zustellung in mm', min_value=0, step=0.1)
        werkzeugtyp = st.text_input('Werkzeugtyp eingeben', '')
        
        # Nutzungszeit des Werkzeugs eingeben
        nutzungszeit = st.number_input('Nutzungszeit des Werkzeugs in Stunden (in halben Schritten)', min_value=0.0, step=0.5)
        
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
            st.write('Klassifizierung:', predicted_class)
            st.write('Vorschub:', vorschub, 'mm/min')
            st.write('Drehzahl:', drehzahl, 'min-1')
            st.write('Zustellung:', zustellung, 'mm')
            st.write('Werkzeugtyp:', werkzeugtyp)
            st.write('Nutzungszeit:', nutzungszeit, 'Stunden')

# Streamlit-App starten
if __name__ == '__main__':
    main()
