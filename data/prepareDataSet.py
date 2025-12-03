import pandas as pd
import numpy as np 
from itertools import chain
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
    
    print("------------------------------")
    print("Optimized dataframe")
    df = optimize(df)
    print(df.head())
    return df  

# new function to optimize the dataset 
def optimize(df):
    # flatten all product tuples
    all_items = list(chain.from_iterable(df["ProductName"]))
    item_to_id = {item: i for i, item in enumerate(sorted(set(all_items)))}

    # map product names to IDs
    df["ProductIDList"] = df["ProductName"].apply(lambda x: [item_to_id[i] for i in x])

    # save transactions with IDs
    df[["TransactionNo", "ProductIDList"]].to_csv("transactions_ids.csv", index=False)

    # save mapping
    pd.DataFrame({"product_id": list(item_to_id.values()), "product_name": list(item_to_id.keys())}).to_csv("product_mapping.csv", index=False)

    return df

read_and_prepare().to_csv("transformed_dataset.csv" , index=False)

