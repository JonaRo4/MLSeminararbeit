import streamlit as st

def main():
    st.title("Einfache Begrüßungsanwendung")

    # Eingabefeld für den Namen des Benutzers
    name = st.text_input("Gib deinen Namen ein:")

    # Überprüfen, ob ein Name eingegeben wurde
    if name:
        st.write(f"Hallo, {name}! Willkommen bei Streamlit.")

if __name__ == "__main__":
    main()
