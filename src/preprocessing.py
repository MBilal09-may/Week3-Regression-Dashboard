#Function to load data
import pandas as pd


def load_data():
    df =pd.read_csv("dataset.csv")
    return df

#Function to clean the df

def cleaned_df(df):
    clean_df = df.copy()
    clean_df = clean_df.drop("Price",axis=1)
    clean_df = clean_df.rename(columns={"Price(USD)":"Price"})
    clean_df["Price"] =round(clean_df["Price"],0) 
    clean_df["Price"]=clean_df["Price"].astype(int)
    clean_df=clean_df.dropna()
    clean_df["Area"]=clean_df["Area"].str.replace(",","")
    clean_df["Area"]=clean_df["Area"].astype(int)
    return clean_df


#Function to encode data

def encoding(input_data,encoder):
    encoded_address = encoder.transform(input_data[["Address"]])
    input_data = input_data.drop("Address",axis=1)
    encoded_address = pd.DataFrame(
        encoded_address,
        columns=encoder.get_feature_names_out(["Address"])
    )
    input_data = pd.concat([input_data.reset_index(drop=True),
                            encoded_address.reset_index(drop=True) ],
                            axis=1)
    return input_data


#Scaling data

def scaling(input_data,scaler):
    scaled_input = scaler.transform(input_data)
    return scaled_input

def preprocess_input(input_data, encoder, scaler):
    
    print(scaling)
    input_data = encoding(input_data, encoder)
    scaled_input = scaling(input_data, scaler)  
    return scaled_input


