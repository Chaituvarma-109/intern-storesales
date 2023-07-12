import streamlit as st

from sales.logger import logging
from sales.pipeline.predict_pipeline import CustomData, PredictPipeline

fat_content = ['Low Fat', 'Regular']
item_type = ['Food', 'Drinks', 'Non-Consumable']
outlet_size = ['Medium', 'Small', 'High']
outlet_location_type = ['Tier 1', 'Tier 2', 'Tier 3']
outlet_type = ['Supermarket Type1', 'Supermarket Type2', 'Grocery Store', 'Supermarket Type3']

st.set_page_config(page_title="Store Sales Prediction App", page_icon="🧊", layout="wide", )

st.header('Prediction Form')
with st.form("my_form"):
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)
    col10, col11 = st.columns(2)

    with col1:
        Item_Identifier = st.text_input('Item_Identifier')

    with col2:
        Item_Weight = st.number_input('Item_Weight')

    with col3:
        Item_Fat_Content = st.selectbox('Item_Fat_Content', fat_content)

    with col4:
        Item_Visibility = st.number_input('Item_Visibility')

    with col5:
        Item_Type = st.selectbox('Item_Type', item_type)

    with col6:
        Item_MRP = st.number_input('Item_MRP')

    with col7:
        Outlet_Identifier = st.text_input('Outlet_Identifier')

    with col8:
        Outlet_Establishment_Year = st.number_input('Outlet_Establishment_Year')

    with col9:
        Outlet_Size = st.selectbox('Outlet_Size', outlet_size)

    with col10:
        Outlet_Location_Type = st.selectbox('Outlet_Location_Type', outlet_location_type)

    with col11:
        Outlet_Type = st.selectbox('Outlet_Type', outlet_type)

    # Every form must have a submit button.
    predict = st.form_submit_button("Predict Item Outlet Sale", type="primary")
    if predict:
        sales_data = CustomData(Item_Identifier=Item_Identifier,
                                Item_Weight=Item_Weight,
                                Item_Fat_Content=Item_Fat_Content,
                                Item_Visibility=Item_Visibility,
                                Item_Type=Item_Type,
                                Item_MRP=Item_MRP,
                                Outlet_Identifier=Outlet_Identifier,
                                Outlet_Establishment_Year=Outlet_Establishment_Year,
                                Outlet_Size=Outlet_Size,
                                Outlet_Location_Type=Outlet_Location_Type,
                                Outlet_Type=Outlet_Type,
                                )

        logging.info("getting sales data as dataframe")
        sales_df = sales_data.get_data_as_data_frame()

        logging.info(">>>>>>>>> Prediction started >>>>>>>>>")

        predict_pipeline = PredictPipeline()

        logging.info("Mid Prediction")
        results = predict_pipeline.predict(sales_df)

        st.metric("Prediction result:", value=results)

        logging.info("after Prediction")
        logging.info(f"results: {results}")
        logging.info(">>>>>>>>> Prediction Completed >>>>>>>>>")
