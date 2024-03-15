import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

# Laden des vortrainierten Modells
model = torch.load('FM_richtig.pth', map_location=torch.device('cpu'))
model.eval()

# Transformation für die Vorverarbeitung der Bilder
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Funktion zur Vorhersage der Werkzeugklasse
def predict_tool_class(image):
    # Vorverarbeitung des Bildes
    image = preprocess(image).unsqueeze(0)

    # Vorhersage mit dem Modell
    with torch.no_grad():
        outputs = model(image)
        predicted_class = F.softmax(outputs, dim=1)

    # Klassenbezeichnungen
    classes = ['Defekt', 'Mittel', 'Neu']
    predicted_class = classes[predicted_class.argmax()]

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

        # Klassifizierung vornehmen, wenn der Benutzer auf den Button klickt
        if st.button('Klassifizieren'):
            tool_class = predict_tool_class(image)
            st.write('Die Werkzeugklasse ist:', tool_class)

if __name__ == '__main__':
    main()



