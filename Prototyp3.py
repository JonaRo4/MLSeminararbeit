import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd

# Laden des trainierten Modells
model = load_model("FM.h5")

# Klassenlabels definieren
class_labels = ['Neu', 'Mittel', 'Defekt']

# Streamlit-App definieren
def main():
    st.title('Werkzeugklassifizierung')
    
    # Spalten für Bild, Dateneingaben und Plot erstellen
    col1, col2, col3 = st.beta_columns([2, 1, 1])
    
    # Bild hochladen
    uploaded_image = col1.file_uploader("Bild von Werkzeug hochladen", type=["jpg", "png", "jpeg"])
    
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
            
            # Ergebnisse anzeigen
            st.write('Werkzeugzustand:', predicted_class)
            
            # Tabelle für die Ergebnisse erstellen
            result_data = {'Werkzeugtyp': [werkzeugtyp],
                           'Nutzungszeit (Stunden)': [nutzungszeit],
                           'Werkzeugzustand': [predicted_class]}
            df = pd.DataFrame(result_data)
            st.table(df)

            # Plot der Eingabedaten
            plot_data(nutzungszeit, predicted_class)

# Funktion zum Plotten der Eingabedaten
def plot_data(nutzungszeit, predicted_class):
    fig, ax = plt.subplots()
    
    # Klassenlabels und ihre Positionen definieren
    labels = ['Neu', 'Mittel', 'Defekt']
    y_pos = np.arange(len(labels))
    
    # Positionen für die Nutzungszeit berechnen
    x_pos = [nutzungszeit] * len(labels)
    
    # Plot erstellen
    ax.barh(y_pos, x_pos, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Klassenlabels umdrehen
    ax.set_xlabel('Nutzungszeit (Stunden)')
    ax.set_title('Eingabedaten')
    
    # Ergebnisse anzeigen
    st.pyplot(fig)

# Streamlit-App starten
if __name__ == '__main__':
    main()



