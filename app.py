import streamlit as st
import pandas as pd
import model
import json
from io import BytesIO
import base64


# ------------------------ Some Caching -----------------------------------
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


st.set_page_config(layout="wide")

# ------------------------------- Actual Frontend ---------------------------------


st.markdown("<h1 style='text-align: center;'>Attrition Model</h1>",
            unsafe_allow_html=True)

powerbi_dashboard = '<iframe title="Attrition dashboard" width="99%" height="800" src="https://app.powerbi.com/reportEmbed?reportId=9295e87d-b608-4144-ae8a-f923884dd11e&autoAuth=true&ctid=65c6160f-c84e-4b96-9faa-8cf33c7be39f" frameborder="0" allowFullScreen="true"></iframe>'
st.markdown(powerbi_dashboard, unsafe_allow_html=True)


st.title('Realtime Prediction')


# Give the user a template file for reference
csv = convert_df(pd.read_csv('./Template_file.csv'))

st.download_button(
    label="Download Template Data",
    data=csv,
    file_name='Template_data.csv',
    mime='text/csv',
)


# Let the user upload a file
file = st.file_uploader("Upload CSV", type=["csv"])

# Check if the user has uploaded a file
if file is not None:
    # Read the CSV file
    df = pd.read_csv(file)
    df = model.filter_data(df)


if st.button(label='Preview Uploaded file'):
    if file is not None:
        st.write(df)
    else:
        st.warning('Please Upload A File')


download_available = False
if st.button(label='Predict Attrition'):
    if file is not None:
        data = model.transorm_data(df=df)
        predicted_data = model.predict_data(data_list_dict=data)
        predicted_data = json.loads(predicted_data)[
            'Results']['WebServiceOutput0']
        output_df = model.output_df_template(predicted_data)
        st.write(output_df)
        output_file = convert_df(output_df)
        download_available = True

    else:
        st.warning('Please Upload A File')


if download_available:
    # Create a Download button for the predicted data
    st.download_button(
        label="Download Predicted Data",
        data=output_file,
        file_name='Predicted_Data.csv',
        mime='text/csv',)
