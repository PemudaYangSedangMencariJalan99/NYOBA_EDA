import streamlit as st
import pandas as pd
import numpy as np

# Judul aplikasi
st.title("Well Test Data Processing")

# Upload file
uploaded_files = st.file_uploader("Choose CSV or Excel files", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    # Membuat dropdown untuk memilih file yang ditunjukkan
    file_names = [uploaded_file.name for uploaded_file in uploaded_files]
    selected_file_name = st.selectbox("Select a file to display", file_names)

    # Menampilkan file yang dipilih
    selected_file = next(uploaded_file for uploaded_file in uploaded_files if uploaded_file.name == selected_file_name)
    
    if selected_file.name.endswith(".csv"):
        df = pd.read_csv(selected_file)
    elif selected_file.name.endswith(".xlsx"):
        df = pd.read_excel(selected_file)

    # Menyimpan dataframe ke session state untuk diakses di EDA1.py
    st.session_state['df'] = df

    # Menampilkan data untuk file yang dipilih
    st.write(f"File Uploaded: {selected_file_name}")
    st.write(df)

    # Dropdown untuk memilih kolom yang digunakan sebagai pemisah file
    column_names = df.columns.tolist()
    selected_column = st.selectbox("Select a column to separate files", column_names)

    # Text input untuk menamai file yang dipilih
    file_name_input = st.text_input("Enter a name for the selected file", value=selected_column)

    # Button untuk memproses file berdasarkan pemisah kolom dan nama file
    if st.button("Process Files"):
        # Menyimpan hasil pemisahan di session_state
        separated_files = {}
        
        # Memproses file berdasarkan kolom yang dipilih
        for idx, file in enumerate(df[selected_column].unique()):
            separated_files[f"well_test_{idx+1}"] = df[df[selected_column] == file].reset_index(drop=True)

        # Simpan hasil pemisahan ke session_state
        st.session_state['separated_files'] = separated_files

        # Menampilkan file yang diproses
        for idx in range(len(df[selected_column].unique())):
            well_test_data = separated_files[f"well_test_{idx+1}"]
            st.write(f"Processed Data for {file_name_input} {idx+1}:")
            st.write(well_test_data)

        st.write("Files have been processed and stored in session state.")
