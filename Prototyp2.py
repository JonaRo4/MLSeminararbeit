import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd

# Laden des trainierten Modells
model = load_model("FM.h5")

# Klassenlabels definieren
class_labels = ['Neu', 'Mittel', 'Defekt']

# Datenrahmen für die Speicherung der Ergebnisse erstellen
results_df = pd.DataFrame(columns=['Bild', 'Werkzeugzustand', 'Nutzungszeit'])

# Streamlit-App definieren
def main():
    st.title('Werkzeugklassifizierung')
    
    # Spalten für Bild und Dateneingaben erstellen
    col1, col2 = st.beta_columns([2, 1])
    
    # Bild hochladen
    uploaded_image = col1.file_uploader("Bild von Werkzeug hochladen", type=["jpg", "png", "jpeg"], key="uploaded_image")
    
    if uploaded_image is not None:
        col1.image(uploaded_image, caption='Hochgeladenes Bild', use_column_width=True)
        
        # Vorschub, Drehzahl, Zustellung und Werkzeugtyp eingeben
        vorschub = col2.number_input('Vorschub in mm/min', min_value=0, step=1)
        drehzahl = col2.number_input('Drehzahl in min-1', min_value=0, step=1)
        zustellung = col2.text_input('Zustellung in mm', value='0.0')
        werkzeugtyp = col2.text_input('Werkzeugtyp eingeben', '')
        
        try:
            zustellung = float(zustellung)
            assert zustellung >= 0  # Überprüfen, dass die Eingabe nicht negativ ist
        except (ValueError, AssertionError):
            col2.error('Bitte geben Sie eine gültige positive Zahl für die Zustellung ein.')
            return
        
        # Nutzungszeit des Werkzeugs eingeben
        nutzungszeit = col2.number_input('Nutzungszeit des Werkzeugs in Stunden', min_value=0.0, step=0.5)
        
        # Klassifizierung des hochgeladenen Bildes
        if col2.button('Klassifizieren'):
            image = Image.open(uploaded_image)
            image = image.resize((224, 224))  # Größe an das Modell anpassen
            
            # Vorverarbeitung des Bildes
            image = np.array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            
            # Vorhersage machen
            prediction = model.predict(image)
            predicted_class = class_labels[np.argmax(prediction)]
            
            # Ergebnisse speichern
            results_df.loc[len(results_df)] = [uploaded_image.name, predicted_class, nutzungszeit]
            
            # Ergebnisse anzeigen
            st.write('Werkzeugzustand:', predicted_class)
    
    # Tabelle der gespeicherten Ergebnisse anzeigen
    st.write('Gespeicherte Ergebnisse:')
    st.write(results_df.astype({'Nutzungszeit': float}))  # Explizite Festlegung des Datentyps


# Streamlit-App starten
if __name__ == '__main__':
    main()

