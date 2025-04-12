import streamlit as st
import pandas as pd
import numpy as np
from scipy import interpolate

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

# Dropdown to select a column for interpolation
column_names = df.columns.tolist()
selected_column = st.selectbox("Select a column for interpolation", column_names)

# Dropdown for interpolation method
interpolation_method = st.selectbox("Select interpolation method", ["Linear", "Polynomial", "Spline"])

# Define interpolation function
def apply_interpolation(df, column, method='linear'):
    # Check for numeric columns for interpolation
    if df[column].dtype not in [np.float64, np.int64]:
        st.error(f"The selected column {column} is not numeric. Please select a numeric column.")
        return df
    
    # Interpolation logic based on method selected
    x = np.arange(len(df))
    y = df[column].values

    if method == 'Linear':
        interpolator = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
    elif method == 'Polynomial':
        interpolator = interpolate.interp1d(x, y, kind='quadratic', fill_value="extrapolate")
    elif method == 'Spline':
        interpolator = interpolate.CubicSpline(x, y, bc_type='natural')
    
    # Apply interpolation
    df[column + "_interpolated"] = interpolator(x)
    return df

# Display the selected data
st.write(f"Data from selected file: {selected_file}")
st.write(df)

# Button to update data based on interpolation method
if st.button("Apply Interpolation"):
    df = apply_interpolation(df, selected_column, interpolation_method)
    st.write("Updated Data with Interpolation Applied:")
    st.write(df)

    # Store the updated dataframe in session state for further use
    st.session_state['df'] = df
