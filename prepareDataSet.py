import pandas as pd
import numpy as np 

url = "dataset.csv"
# read the data set into a dataframe
def read_and_prepare():
    df = pd.read_csv(url)
    print("raw data :")
    print(df.head(10))
    print(df.shape)
    print(df.columns)
    print("------------------------------")
    df = df.groupby("TransactionNo")["ProductName"].apply(lambda x:tuple(dict.fromkeys(x))).reset_index() 
    print("treansformed data")
    print(df.head())
    print(df.shape)
    print(df.columns)
    return df 
read_and_prepare().to_csv("transformed_dataset.csv" , index=False)
