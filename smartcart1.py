import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("smartcart_customers.csv")
df.head()
df.isnull().sum()
#data preprocessing
#handlemissing val
df["Income"]=df["Income"].fillna(df["Income"].median())
df.isnull().sum()
#feature engi
#for age
df["age"]=2026-df["Year_Birth"]
#customerjoining date
df["Dt_Customer"]=pd.to_datetime(df["Dt_Customer"],dayfirst=True)

reference_date=df["Dt_Customer"].max()

df["Customer_Tenure_Days"]=(reference_date-df["Dt_Customer"]).dt.days
