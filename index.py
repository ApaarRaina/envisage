import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv('Haryana_data.csv')
row_data = df.iloc[0, :]
columns = df.columns
df1 = pd.DataFrame([row_data], columns=columns)

with open('model_pickle', 'rb') as f:
    dt = pickle.load(f)

# Streamlit UI
st.image("https://www.shutterstock.com/image-photo/lush-rice-paddy-field-neat-600nw-2499404003.jpg", width=1600)
st.title(":green[Crop Yield Predictor] :seedling:")
st.markdown("*This Predictor is only applicable to the crops grown in the state of Haryana.*")

with st.form(key="my_form"):
    state = st.selectbox("State", ['Haryana'])
    districts = [
        "Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad",
        "Gurugram", "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal",
        "Kurukshetra", "Mahendragarh", "Nuh", "Palwal", "Panchkula",
        "Panipat", "Rewari", "Rohtak", "Sirsa", "Sonipat", "Yamunanagar"
    ]
    district = st.selectbox("District Name", options=districts)

    current_year = 2025
    selected_year = st.selectbox("Select Year", options=range(2014, current_year + 1))

    crops = [
        "Arhar/Tur", "Bajra", "Banana", "Barley", "Castor seed",
        "Coriander", "Cotton(lint)", "Dry chillies", "Dry ginger", "Garlic",
        "Gram", "Grapes", "Groundnut", "Guar seed", "Horse-gram", "Jowar",
        "Maize", "Mango", "Masoor", "Moong(Green Gram)", "Moth", "Onion",
        "Rabi pulses", "Peas & beans (Pulses)", "Potato", "Rapeseed & Mustard",
        "Rice", "Sannhamp", "Sesamum", "Sugarcane", "Sunflower", "Sweet potato",
        "Turmeric", "Urad", "Wheat", "Other", "Other Fresh Fruits", "Other Kharif pulses",
        "Other Vegetables"
    ]
    crop_type = st.selectbox("Crop Type", options=crops)
    season = st.selectbox("Season", ["Whole Year", "Kharif", "Rabi"])
    area = st.number_input("Area of Land (in hectares)", min_value=0.0, max_value=100000.0, step=0.1)
    submit_button = st.form_submit_button("Predict Yield")

    if submit_button:
        # Preprocess inputs
        district = district.replace(" ", "").upper()
        state = state.replace(" ", "")
        crop_type = crop_type.replace(" ", "")
        season = season.replace(" ", "")
        selected_year = str(selected_year)

        # Standardize column names
        df1.columns = df1.columns.str.replace(" ", "")

        # Match input with training features
        expected_features = dt.feature_names_in_
        for feature in expected_features:
            if feature not in df1.columns:
                df1[feature] = 0
        df1 = df1[expected_features]

        # Update input values
        if crop_type in df1.columns:
            df1[crop_type] = 1
        if season in df1.columns:
            df1[season] = 1
        if state in df1.columns:
            df1[state] = 1
        if district in df1.columns:
            df1[district] = 1
        if 'Year' in df1.columns:
            df1['Year'] = selected_year
        if 'Area' in df1.columns:
            df1['Area'] = area

        # Predict yield
        y_pred = dt.predict(df1)

        predicted_yield = []
        date = []
        for i in range(2015, 2026):
            df1['Year'] = i
            df1[state] = 1
            df1[crop_type] = 1
            df1['Area'] = area
            df1[district] = 1
            df1[season] = 1
            if area!=0:
                put=dt.predict(df1)
            else:
                put=0
            predicted_yield.append(put)
            date.append(i)

        fig=plt.figure(figsize=(10, 10))

        plt.scatter(date, predicted_yield)
        plt.xlabel("The date")
        plt.ylabel("The predicted yield")
        plt.show()

        st.pyplot(fig)
        st.write(df1)

        if area==0:
            y_pred=[0]

        # Display prediction
        st.write(f"The predicted yield is {y_pred[0]}")


with st.expander("About"):
    st.write("""
    *Crop Yield Predictor* is a tool to predict crop yield based on various factors such as location, crop type, and season. 
    It uses machine learning models to provide accurate predictions based on past agricultural data.
    """)