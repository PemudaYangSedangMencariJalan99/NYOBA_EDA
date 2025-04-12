import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import LabelEncoder

# Load session state data
if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("Please upload a file first.")
    st.stop()

# Check if there are processed files
if 'separated_files' not in st.session_state or not st.session_state['separated_files']:
    st.warning("No processed files available. Please process the files first.")
    st.stop()

# Dropdown to select the file for analysis
file_options = ["Uploaded File"] + list(st.session_state['separated_files'].keys())
selected_file = st.selectbox("Select a file to analyze", file_options)

# Load the selected file
if selected_file == "Uploaded File":
    df = st.session_state['df']
else:
    df = st.session_state['separated_files'][selected_file]

# Dropdown to select columns for scaling
column_names = df.columns.tolist()
selected_columns = st.multiselect("Select columns for scaling", column_names)

# Dropdown for transformation method
scaling_methods = ["Min-Max Scaling", "Box-Cox", "Yeo-Johnson"]
selected_method = st.selectbox("Select feature scaling method", scaling_methods)

# LabelEncoder function
def label_encode(df, columns):
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df

# Apply transformations function
def apply_scaling(df, selected_columns, method='Min-Max Scaling'):
    if not selected_columns:
        st.error("No columns selected for transformation.")
        return df

    # Apply Label Encoding for categorical columns if selected
    if 'Label Encoding' in selected_columns:
        df = label_encode(df, selected_columns)

    # Apply the selected transformation method
    if method == 'Min-Max Scaling':
        scaler = MinMaxScaler()
        df[selected_columns] = scaler.fit_transform(df[selected_columns])

    elif method == 'Box-Cox':
        # Box-Cox requires positive values, so shift data by adding a small constant if necessary
        for col in selected_columns:
            df[col] = df[col] + 1  # Ensuring positive values for Box-Cox
            df[col], _ = boxcox(df[col])

    elif method == 'Yeo-Johnson':
        for col in selected_columns:
            df[col], _ = yeojohnson(df[col])

    return df

# Display the selected data
st.write(f"Data from selected file: {selected_file}")
st.write(df)

# Button to apply the selected transformation
if st.button("Apply Transformation"):
    df = apply_scaling(df, selected_columns, selected_method)
    st.write(f"Data after {selected_method}:")
    st.write(df)

    # Store the updated dataframe in session state for further use
    st.session_state['df'] = df
