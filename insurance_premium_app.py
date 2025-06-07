import streamlit as st
import pandas as pd
import joblib

# Load model, encoders, and training column names
rf_model = joblib.load("rf_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Car Insurance Premium Predictor")

# Use encoder class lists to populate dropdowns
#INSR_TYPE_options = label_encoders['INSR_TYPE'].classes_
MAKE_options = label_encoders['MAKE'].classes_
USAGE_options = label_encoders['USAGE'].classes_

# UI inputs
SEX = st.selectbox("Sex", [0, 1, 2])
EFFECTIVE_YR = st.number_input("Effective Year", min_value=0, max_value=2025)
INSR_TYPE = st.selectbox("Insurance Type", [1201, 1202, 1204])
INSURED_VALUE = st.number_input("Insured Value", min_value=0)
MAKE = st.selectbox("Make", ["NISSAN", "TOYOTA",
                             "IVECO", "FIAT", "MARU", "ISUZU",
                             "YAMAHA", "SUZUKI", "MITSUBISHI",
                             "MESFIN", "CALABRASE", "DAF", "FARID",
                             "BMW", "ISUSU", "ZZ", "FORD", "PEUGEOT",
                             "TOMSON", "MERCEDES", "MERCEEDES",
                             "VIBERTI", "HOWO", "DAEWOO", "ROZA",
                             "VOLKSWAGON", "CRANE", "TURBO BUS", "SCANIA",
                             "NATFA", "RENAULT", "INTERNATIONAL USE",
                             "MAZDA", "NEW HOLLAND", "NEW HOLAND",
                             "ROLFO", "MACK", "MTE", "ORAL", "ZONGUSHEN",
                             "HYUNDAI", "ABAY", "NAMI", "VOLVO", "BISHOFTU",
                             "DATSUN", "TRAKKER", "AEOLUS",
                             "TATA", "HINO", "CALABRESE", "OPEL",
                             "VERYCA", "NISSAN UD", "TURBO", "TRAILER",
                             "CHEVROLET", "LOWBED", "MOTOR CYCLE (TWOCYCLE)",
                             "HONDA", "HIGER BUS", "HIGHER",
                             "LANDROVER", "DACIA", "KIA", "GEEP",
                             "ZORZI", "CACCIAMALLI", "BELARUS", "RANDON", "CORDES", 
                             "BAJAJI", "HIGH BED TRAILER", "JEEP", "ILSBO",
                             "SKODA", "ASTRA", "FOTON", "AUDI", "INTERNATIONAL", " DONG FENG",
                             "LADA", "SINO HOWO", "BARTOLETI", "ADGE", "BOXER", 
                             "DAMAS", "BELARUS TRACTOR", "WHEEL LOADER", "MERCEEDICE",
                             "NIVA", "MAHINDRA", "TRUCK", "OTOYOL", "DAIHATSU",
                             "PREGIO", "FAW", "VITZ", "MAZ", "ME", "TVS", "BAJAJ",
                             "LIFAN", "VOLKS WAGON", "SCHMITZ", "GAME", "KYRON",
                             "DOCC", "CHERRY", "ATOZ", "HIGER", "DAWOO", " HERO", "SPORTAGE",
                             "ZX-TOP", "LAND CRUISER", "SUGERCANE TRAILER", "GLEEY",
                             "MERCHEDES", "COMBI", "CORE DRILLING RING", "WATER PUMP", 
                             "JIN BEI", "SOCOOL", "RANGEROVER", "JAC", "LISBO", "DIAHATSU",
                             "SINO", "BELL TRACTOR", "ZONGSHEN", "CADILLAC", "TOYOTA*", "CLASS",
                             "GMC", "CALDINESS FORK LIFTS", "DAHATSUN", "DAIHATSU TERIOS",
                             "RENAULT*", "MITSUBISHI*", "NISSAN*", "T0Y0TA", "SKY BUS", "KAMAZ",
                             "KAMZ", "GRADER", "EXCAVATOR", "LOADER", "DOZER", "LAND ROVER", "TRAKER", "MINI BUS", "LEXUS",
                             "MAHANDRA", "PLATENA", "CATERPILLAR", "BRIDGE", "REXTON", "AWASH", "ROLLER", "VOLKSWAGEN", 
                             "CHANA", "1982", "AUTO", "WINEGEL", "P/UP", "BEBEN SEMI TRAILER", "JOHN DEER", "JOHNDEER", "BELL", "GEELY",
                             "AMBULANCE", "FOTTON", "RENAULT/STOLARCZYK", "TRACTOR TRAILER", "330-30 TRAILER", "MAN", "MASSY FUREGUSON",
                             "SAMI", "CITROEN", "RED FOX", "CAT DOZER", "ZEPPLIN", "VIBRATION ROLLER", "ZOTYE", "CATERPILLAR TRACTOR",
                             "TRACTOR", "CASE", "VERCYA", "KAT", "FORCE", "TICO", "JIEFANG", "LANDINI", "TALER", "ZMAY",
                             "DANDO GEATECH 7.5 HYDOLIC TOP ROTATYING", "GREAT WALL", "VERSATILE", "POWER PLUS DOSER", "POWER PLUS DOZER", 
                             "NIO", "HOVER", "WUCING", "ISUZU FVR", "CHANGHE", "USA", "FORLAND", "DIHATSU", "EMGTAND", "CAT", "GORICA",
                             "AFRO", "PAGOT", "PONTIAC", "TOYOTA LAND CRUISER", "1985", "MATIZ", "SUZUKI GRAND VITARA", "DANGIFAN", "ALFA",
                             "MOTOR CYCLE", "LONGJIANG", "LOW BED", "VISTO", "VOLS WAGEN", "IFA", "DAYUN", "TERIOS", "DISCOVERY2.5", "TEKEZE",
                             "CAMZ", "2011", "ZUNGSHUN", "WAZ"])
USAGE = st.selectbox("Usage", ["Own Goods", "General Cartage", "Fare Paying Passengers",
                               "Private", "Own service", "Taxi", "Agricultural Own Farm",
                               "Special Construction", "Learnes", "Car Hires",
                               "Ambulance", "Agricultural Any Farm", "Others",
                               "Fire fighting"])
CLAIM_PAID_FLAG = st.selectbox("Claim Paid Flag", [0, 1])
PROD_YEAR = st.number_input("Production Year", min_value=1900, max_value=2025)
SEATS_NUM = st.number_input("Number of Seats", min_value=1, max_value=100)
CARRYING_CAPACITY = st.number_input("Carrying Capacity", min_value=0.0)
VEHICLE_TYPE = st.selectbox("Vehicle Type", ["Pick-up", "Truck", "Bus", "Trailers and semitrailers", "Automobile",
                                             "Motor-cycle", "Station Wagones", "Tractor", "Special construction", "Tanker"])

# Construct input DataFrame
X_input = pd.DataFrame({
    'SEX': [SEX],
    'EFFECTIVE_YR': [EFFECTIVE_YR],
    'INSR_TYPE': [INSR_TYPE],
    'INSURED_VALUE': [INSURED_VALUE],
    'MAKE': [MAKE],
    'USAGE': [USAGE],
    'CLAIM_PAID_FLAG': [CLAIM_PAID_FLAG],
    'PROD_YEAR': [PROD_YEAR],
    'SEATS_NUM': [SEATS_NUM],
    'CARRYING_CAPACITY': [CARRYING_CAPACITY],
    'VEHICLE_TYPE': [VEHICLE_TYPE]
})

# Encode using saved LabelEncoders
for col in ['MAKE', 'USAGE']:
    le = label_encoders[col]
    try:
        X_input[col] = le.transform(X_input[col])
    except ValueError:
        st.error(f"Invalid input: '{X_input[col].values[0]}' is not recognized in {col}.")
        st.stop()

# Ensure correct column order
X_input = X_input.reindex(columns=model_columns, fill_value=0)

# Predict and display
if st.button("Predict Premium"):
    premium_pred = rf_model.predict(X_input)
    st.success(f"Predicted Premium: {premium_pred[0]:.2f}")