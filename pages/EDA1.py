import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import plotly.graph_objs as go
from matplotlib.ticker import LinearLocator, MultipleLocator, AutoLocator
import pandas as pd

# Initialize session state if not already done
if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'separated_files' not in st.session_state:
    st.session_state['separated_files'] = {}

# Menambahkan CSS untuk mengatur alignment ke kiri
st.markdown("""
    <style>
        .css-1v3fvcr {
            text-align: left;
        }
        .stApp {
            text-align: left;
        }
    </style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.title("Exploratory Data Analysis (EDA)")

# Lithology dictionary
lithology_numbers = {
    30000: {'lith':'Sandstone', 'lith_num':1, 'hatch': '..', 'color':'#ffff00'},
    65030: {'lith':'Sandstone/Shale', 'lith_num':2, 'hatch':'-.', 'color':'#ffe119'},
    65000: {'lith':'Shale', 'lith_num':3, 'hatch':'--', 'color':'#bebebe'},
    80000: {'lith':'Marl', 'lith_num':4, 'hatch':'', 'color':'#7cfc00'},
    74000: {'lith':'Dolomite', 'lith_num':5, 'hatch':'-/', 'color':'#8080ff'},
    70000: {'lith':'Limestone', 'lith_num':6, 'hatch':'+', 'color':'#80ffff'},
    70032: {'lith':'Chalk', 'lith_num':7, 'hatch':'..', 'color':'#80ffff'},
    88000: {'lith':'Halite', 'lith_num':8, 'hatch':'x', 'color':'#7ddfbe'},
    86000: {'lith':'Anhydrite', 'lith_num':9, 'hatch':'', 'color':'#ff80ff'},
    99000: {'lith':'Tuff', 'lith_num':10, 'hatch':'||', 'color':'#ff8c00'},
    90000: {'lith':'Coal', 'lith_num':11, 'hatch':'', 'color':'black'},
    93000: {'lith':'Basement', 'lith_num':12, 'hatch':'-|', 'color':'#ef138a'}
}

# Cek apakah data sudah diupload di main.py
if 'df' in st.session_state and st.session_state['df'] is not None:
    df = st.session_state['df']

    # Dropdown untuk memilih apakah ingin menganalisis file utama atau hasil pemisahan
    file_option = st.selectbox("Select file to analyze", ["Uploaded File", "Processed Files"])

    if file_option == "Uploaded File":
        # Menampilkan data yang di-upload
        st.write(f"Analyzing the uploaded file:")
        st.write(df)
        data_to_analyze = df
        depth_col = st.selectbox("Select depth column", df.columns)
    else:
        # Cek apakah ada separated_files di session_state
        if 'separated_files' in st.session_state and st.session_state['separated_files']:
            # Mendapatkan nama-nama file yang diproses
            processed_file_names = list(st.session_state['separated_files'].keys())
            selected_processed_file = st.selectbox("Select a processed file", processed_file_names)
            
            # Mengambil data dari session_state
            data_to_analyze = st.session_state['separated_files'][selected_processed_file]
            
            st.write(f"Analyzing processed file: {selected_processed_file}")
            st.write(data_to_analyze)
            depth_col = st.selectbox("Select depth column", data_to_analyze.columns)
        else:
            st.warning("No processed files available. Please process files in the main page first.")
            st.stop()
    
    # 1. Descriptive Statistics Section
    st.header("Descriptive Statistics")
    st.write(data_to_analyze.describe())
    
    # 2. Seaborn Gaussian Distribution Section
    st.header("Gaussian Distribution for Each Parameter")
    
    # Select numeric columns only
    numeric_cols = data_to_analyze.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Allow user to select which parameters to view
    selected_params = st.multiselect("Select parameters for distribution plots:", 
                                    options=numeric_cols,
                                    default=numeric_cols[:min(5, len(numeric_cols))])
    
    if selected_params:
        # Create distribution plots with Seaborn
        for i in range(0, len(selected_params), 3):  # Show 3 plots per row
            cols = st.columns(min(3, len(selected_params) - i))
            for j, col in enumerate(cols):
                if i + j < len(selected_params):
                    param = selected_params[i + j]
                    with col:
                        st.subheader(f"Distribution of {param}")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.histplot(data_to_analyze[param].dropna(), kde=True, ax=ax)
                        plt.title(f"Distribution of {param}")
                        plt.xlabel(param)
                        plt.ylabel("Frequency")
                        st.pyplot(fig)
    
    # 3. Plotly Scatterplot Matrix
    st.header("Scatterplot Matrix (Correlation Analysis)")
    
    # Select numeric columns for the scatterplot matrix
    scatter_cols = st.multiselect(
        "Select parameters for the scatterplot matrix (limit to 10 for performance):",
        options=numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )
    
    if scatter_cols:
        if len(scatter_cols) > 10:
            st.warning("Too many parameters selected. For better performance, limiting to first 10.")
            scatter_cols = scatter_cols[:10]
        
        # Normalize data using Min-Max scaling for the selected columns
        data_norm = data_to_analyze.copy()
        for col in scatter_cols:
            min_val = data_to_analyze[col].min()
            max_val = data_to_analyze[col].max()
            if max_val > min_val:  # Avoid division by zero
                data_norm[col] = (data_to_analyze[col] - min_val) / (max_val - min_val)
        
        # Create dimensions for the splom plot
        dimensions = [dict(label=col, values=data_norm[col]) for col in scatter_cols]
        
        # Create Scatterplot Matrix (Splom)
        fig = go.Figure(data=go.Splom(
            dimensions=dimensions,
            marker=dict(size=5, line=dict(width=0.5)),
            diagonal=dict(visible=True),  # Show histogram on diagonal
            showupperhalf=False  # Only show lower half of matrix
        ))
        
        # Update layout
        fig.update_layout(
            title="Distribution Plot of Well Parameters",
            dragmode='select',
            width=900,
            height=900,
            hovermode='closest'
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

    # 4. Well Log Visualization Functions
    st.header("Lithology and Logs Visualization")
    
    # Function for plotting without lithology
    def plot_lithology_and_logs(df, depth_col, log_colors, title):
        # Check if required columns exist in the dataframe
        required_logs = ["GR", "CALI", "RHOB", "NPHI", "RDEP", "DTC", "RMED", "DRHO"]
        missing_logs = [log for log in required_logs if log not in df.columns]
        
        if missing_logs:
            st.warning(f"Missing columns in data: {', '.join(missing_logs)}. Some plots may not display correctly.")
        
        available_logs = [log for log in ["GR", "CALI", "RHOB_NPHI", "RDEP", "DTC", 'RMED', 'DRHO'] 
                          if log == "RHOB_NPHI" or log in df.columns]
        
        if not available_logs:
            st.error("No valid log columns found for plotting.")
            return
            
        fig, axes = plt.subplots(nrows=1, ncols=len(available_logs), figsize=(10, 25))
        
        # Handle case with only one log
        if len(available_logs) == 1:
            axes = [axes]

        for i, log in enumerate(available_logs):
            ax_log = axes[i]

            if log == "GR" and "GR" in df.columns:
                ax_log.plot(df["GR"], df[depth_col], color=log_colors[i % len(log_colors)], linewidth=1)
                if 75 <= df["GR"].max():  # Only fill if the GR range includes 75
                    ax_log.fill_betweenx(df[depth_col], df['GR'], 75, 
                                        where=[gr <= 75 for gr in df['GR']], 
                                        interpolate=True, color='yellow', alpha=0.7)
                    ax_log.fill_betweenx(df[depth_col], df['GR'], 75, 
                                        where=[gr >= 75 for gr in df['GR']], 
                                        interpolate=True, color='green', alpha=0.7)
                ax_log.set_xlim(0, max(200, df["GR"].max() * 1.1))

            elif log == "RDEP" and "RDEP" in df.columns:
                ax_log.set_xscale('log')
                ax_log.plot(df["RDEP"], df[depth_col], color=log_colors[i % len(log_colors)], linewidth=1)

            elif log == "RHOB_NPHI" and "RHOB" in df.columns and "NPHI" in df.columns:
                ax3 = ax_log
                ax4 = ax3.twiny()
                ax3.plot(df["RHOB"], df[depth_col], color='red', linewidth=0.8, label="RHOB")
                ax4.plot(df["NPHI"], df[depth_col], color='blue', linewidth=0.8, linestyle='dashed', label="NPHI")
                ax3.set_xlim(1.65, 2.65)
                ax4.set_xlim(0.6, 0)
                
                x1 = df["RHOB"]
                x2 = df["NPHI"]
                x = np.array(ax3.get_xlim())
                z = np.array(ax4.get_xlim())
                nz = ((x2 - np.max(z)) / (np.min(z) - np.max(z))) * (np.max(x) - np.min(x)) + np.min(x)
                ax3.fill_betweenx(df[depth_col], x1, nz, where=x1 >= nz, interpolate=True, color='green', alpha=0.7)
                ax3.fill_betweenx(df[depth_col], x1, nz, where=x1 <= nz, interpolate=True, color='yellow', alpha=0.7)
                
                ax3.legend()

            elif log == "DTC" and "DTC" in df.columns:
                ax_log.plot(df["DTC"], df[depth_col], color='purple', linewidth=2, label="DTC")
                ax_log.set_xlim(0, 200)
                ax_log.legend()

            elif log == "CALI" and "CALI" in df.columns:
                ax_log.plot(df["CALI"], df[depth_col], color=log_colors[i % len(log_colors)], linewidth=2)
                ax_log.set_xlim(6, 26)
                
            elif log == "RMED" and "RMED" in df.columns:
                ax_log.plot(df["RMED"], df[depth_col], color=log_colors[i % len(log_colors)], linewidth=2)
                ax_log.set_xscale('log')
                
            elif log == "DRHO" and "DRHO" in df.columns:
                ax_log.plot(df["DRHO"], df[depth_col], color=log_colors[i % len(log_colors)], linewidth=2)

            ax_log.set_title(log, fontsize=5)
            ax_log.set_ylim(df[depth_col].max(), df[depth_col].min())
            ax_log.grid(axis='x', which='major', linestyle='-', linewidth=1.5, color='black')
            ax_log.grid(axis='y', which='minor', linestyle='-', linewidth=1, color='black')
            ax_log.grid(axis='y', which='major', linestyle='-', linewidth=1.5, color='black')

            if log == "RDEP" or log == "RMED":
                ax_log.xaxis.set_major_locator(AutoLocator())  
            else:
                ax_log.xaxis.set_major_locator(LinearLocator(numticks=5))  
            ax_log.yaxis.set_major_locator(MultipleLocator(100))  
            ax_log.yaxis.set_minor_locator(MultipleLocator(25))  
            ax_log.tick_params(axis='y', which='both', labelsize=5)  
            ax_log.tick_params(axis='x', which='both', labelsize=5)  

            if log == available_logs[0]:  # Only first plot should show y labels
                ax_log.tick_params(axis='y', which='both', left=True, right=True, labelleft=True)
            else:
                ax_log.tick_params(axis='y', which='both', left=False, right=False, labelleft=False) 

        fig.suptitle(title, fontsize=5, fontweight='bold', y=1.02)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=0.0) 
        
        # Create output folder if it doesn't exist
        output_folder = "well_TEST_log_images"  
        os.makedirs(output_folder, exist_ok=True)  

        # Save figure
        save_path = os.path.join(output_folder, f"{title.replace(' ', '_')}.png")  
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        return fig

    # Function for plotting with lithology
    def plot_lithology_and_logs2(df, depth_col, lithology_col, lithology_numbers, log_colors, title):
        logs = ["GR", "CALI", "GR_NEW", "RHOB_NPHI", "RDEP", "DTC", 'RMED', 'DRHO', 'Lithology_code']
        
        # Check which logs are available in the dataframe
        available_logs = [log for log in logs if (log == "RHOB_NPHI" and "RHOB" in df.columns and "NPHI" in df.columns) or 
                          (log != "RHOB_NPHI" and log in df.columns) or 
                          (log == 'Lithology_code')]
        
        if len(available_logs) < 2:  # At least one log and lithology
            st.error("Not enough valid columns for plotting with lithology.")
            return None
            
        fig, axes = plt.subplots(nrows=1, ncols=len(available_logs), figsize=(20, 30))
        
        # Handle case with only two plots (one log + lithology)
        if len(available_logs) == 2:
            axes = [axes[0], axes[1]]

        for i, log in enumerate(available_logs[:-1]):  # All except lithology
            ax_log = axes[i]

            if log == "GR" and "GR" in df.columns:
                ax_log.plot(df["GR"], df[depth_col], color=log_colors[i % len(log_colors)], linewidth=0.5)
                ax_log.set_xlim(df["GR"].min(), df["GR"].max())
                ax_log.fill_betweenx(df[depth_col], df['GR'], 75, 
                                    where=[gr <= 75 for gr in df['GR']], 
                                    interpolate=True, color='yellow', alpha=0.7)
                ax_log.fill_betweenx(df[depth_col], df['GR'], 75, 
                                    where=[gr >= 75 for gr in df['GR']], 
                                    interpolate=True, color='green', alpha=0.7)
                ax_log.set_xlim(0, 200)
            
            elif log == "GR_NEW" and "GR_NEW" in df.columns:
                ax_log.plot(df["GR_NEW"], df[depth_col], color=log_colors[i % len(log_colors)], linewidth=0.5)
                ax_log.set_xlim(df["GR_NEW"].min(), df["GR_NEW"].max())
                ax_log.set_xlim(0, 200)

            elif log == "RDEP" and "RDEP" in df.columns:
                ax_log.set_xscale('log')
                ax_log.plot(df["RDEP"], df[depth_col], color=log_colors[i % len(log_colors)], linewidth=0.5)

            elif log == "RHOB_NPHI" and "RHOB" in df.columns and "NPHI" in df.columns:
                ax3 = ax_log
                ax4 = ax3.twiny()
                ax3.plot(df["RHOB"], df[depth_col], color='red', linewidth=0.8, label="RHOB")
                ax4.plot(df["NPHI"], df[depth_col], color='blue', linewidth=0.8, linestyle='dashed', label="NPHI")
                ax3.set_xlim(1.65, 2.65)
                ax4.set_xlim(0.6, 0)
                
                x1 = df["RHOB"]
                x2 = df["NPHI"]
                x = np.array(ax3.get_xlim())
                z = np.array(ax4.get_xlim())
                nz = ((x2 - np.max(z)) / (np.min(z) - np.max(z))) * (np.max(x) - np.min(x)) + np.min(x)
                ax3.fill_betweenx(df[depth_col], x1, nz, where=x1 >= nz, interpolate=True, color='green', alpha=0.7)
                ax3.fill_betweenx(df[depth_col], x1, nz, where=x1 <= nz, interpolate=True, color='yellow', alpha=0.7)
                
                ax_log.legend()

            elif log == "DTC" and "DTC" in df.columns:
                ax_log.plot(df["DTC"], df[depth_col], color='purple', linewidth=0.7, label="DTC")
                ax_log.set_xlim(0, 200)
                ax_log.legend()

            elif log == "DTS" and "DTS" in df.columns:
                ax_log.plot(df["DTS"], df[depth_col], color='orange', linewidth=0.7, linestyle='dashed', label="DTS")
                ax_log.set_xlim(0, 200)
                ax_log.legend()

            elif log == "CALI" and "CALI" in df.columns:
                ax_log.plot(df["CALI"], df[depth_col], color=log_colors[i % len(log_colors)], linewidth=0.7)
                ax_log.set_xlim(6, 26)
                
            elif log == "RMED" and "RMED" in df.columns:
                ax_log.plot(df["RMED"], df[depth_col], color=log_colors[i % len(log_colors)], linewidth=0.7)
                ax_log.set_xscale('log')
                
            elif log == "DRHO" and "DRHO" in df.columns:
                ax_log.plot(df["DRHO"], df[depth_col], color=log_colors[i % len(log_colors)], linewidth=0.7)

            ax_log.set_title(log, fontsize=5)
            ax_log.set_ylim(df[depth_col].max(), df[depth_col].min())
            ax_log.grid(axis='x', which='major', linestyle='-', linewidth=0.7, color='black')
            ax_log.grid(axis='y', which='minor', linestyle='-', linewidth=0.4, color='black')
            ax_log.grid(axis='y', which='major', linestyle='-', linewidth=0.4, color='black')

            if log == "RDEP" or log == "RMED":
                ax_log.xaxis.set_major_locator(AutoLocator())  
            else:
                ax_log.xaxis.set_major_locator(LinearLocator(numticks=5))  
            ax_log.yaxis.set_major_locator(MultipleLocator(100))  
            ax_log.yaxis.set_minor_locator(MultipleLocator(25))  
            ax_log.tick_params(axis='y', which='both', labelsize=1)  
            ax_log.tick_params(axis='x', which='both', labelsize=1)  

        # Lithology plot (last column)
        ax_lith = axes[-1]
        
        # Check if lithology column exists
        if lithology_col in df.columns:
            ax_lith.plot(df[lithology_col], df[depth_col], color="black", linewidth=0.5)
            ax_lith.set_xlabel("Lithology", fontsize=5, color="black")
            ax_lith.yaxis.set_minor_locator(MultipleLocator(50))
            ax_lith.set_xlim(0, 1)

            # Fill lithology column based on codes
            for key in lithology_numbers.keys():
                color = lithology_numbers[key]['color']
                hatch = lithology_numbers[key]['hatch']
                if key in df[lithology_col].unique():
                    ax_lith.fill_betweenx(df[depth_col], 0, 1, 
                                        where=(df[lithology_col] == key),
                                        facecolor=color, hatch=hatch)

            ax_lith.set_xticks([0, 1])
            ax_lith.set_title('LITHOLOGY', fontsize=5)
            ax_lith.set_ylim(df[depth_col].max(), df[depth_col].min())
        else:
            ax_lith.set_title('LITHOLOGY (No Data)', fontsize=5)
            ax_lith.set_ylim(df[depth_col].max(), df[depth_col].min())
            
        fig.suptitle(title, fontsize=5, fontweight='bold', y=1.02)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=0.0) 
        output_folder = "well_log_with_lithology_images"  
        os.makedirs(output_folder, exist_ok=True)  

        save_path = os.path.join(output_folder, f"{title.replace(' ', '_')}_with_lithology.png")  
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        return fig
    
    # Add dropdown to select visualization type
    visualization_type = st.selectbox(
        "Select visualization type:",
        ["Without Lithology", "With Lithology"]
    )
    
    # If "With Lithology" is selected, add column selector for lithology
    if visualization_type == "With Lithology":
        lithology_col = st.selectbox(
            "Select lithology column:",
            data_to_analyze.columns,
            index=min(8, len(data_to_analyze.columns)-1)  # Default to a reasonable column index
        )
        
        # Preview lithology codes in the dataset
        if lithology_col in data_to_analyze.columns:
            unique_codes = data_to_analyze[lithology_col].unique()
            available_codes = [code for code in unique_codes if code in lithology_numbers]
            
            if available_codes:
                st.write("Available lithology codes in dataset:")
                lithology_info = []
                for code in available_codes:
                    lithology_info.append({
                        "Code": code,
                        "Lithology": lithology_numbers[code]['lith'],
                        "Color": lithology_numbers[code]['color']
                    })
                st.table(pd.DataFrame(lithology_info))
            else:
                st.warning(f"No matching lithology codes found in the selected column. Available values: {unique_codes}")
                st.write("Expected codes must match the lithology dictionary keys.")
    
    # Add a button to generate the plot
    if st.button("Generate Log Visualization"):
        try:
            if visualization_type == "Without Lithology":
                plot_lithology_and_logs(
                    data_to_analyze, 
                    depth_col=depth_col, 
                    log_colors=["red", "blue", "green", "purple", "orange", "yellow", "pink"], 
                    title="Well Log Analysis"
                )
                st.success(f"Plot saved to well_TEST_log_images folder")
            else:  # With Lithology
                if lithology_col not in data_to_analyze.columns:
                    st.error(f"Selected lithology column '{lithology_col}' not found in data.")
                else:
                    plot_lithology_and_logs2(
                        data_to_analyze, 
                        depth_col=depth_col,
                        lithology_col=lithology_col,
                        lithology_numbers=lithology_numbers,
                        log_colors=["red", "blue", "green", "purple", "orange", "yellow", "pink"], 
                        title="Well Log Analysis With Lithology"
                    )
                    st.success(f"Plot saved to well_log_with_lithology_images folder")
        except Exception as e:
            st.error(f"Error generating plot: {str(e)}")
            st.error("Traceback:")
            import traceback
            st.code(traceback.format_exc())

else:
    st.write("No data available. Please upload data in the main page first.")