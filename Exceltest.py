import streamlit as st
import pandas as pd
pip install openpyxl


# Beispiel Datenframe
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda'],
    'Alter': [28, 35, 40, 25],
    'Stadt': ['Berlin', 'Hamburg', 'München', 'Frankfurt']
}

df = pd.DataFrame(data)

# Streamlit App
st.write(df)

# Button zum Speichern in Excel
if st.button('Daten in Excel speichern'):
    # Dateiname
    excel_file = 'daten.xlsx'
    # Schreibe Daten in Excel
    df.to_excel(excel_file, index=False)
    st.success(f'Daten wurden in {excel_file} gespeichert.')
