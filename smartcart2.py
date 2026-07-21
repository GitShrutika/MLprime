#customerjoining date
df["Dt_Customer"]=pd.to_datetime(df["Dt_Customer"],dayfirst=True)

reference_date=df["Dt_Customer"].max()

df["Customer_Tenure_Days"]=(reference_date-df["Dt_Customer"]).dt.days

df["Total_Spending"]=df["MntWines"]+df["MntFruits"]+df["MntMeatProducts"]+df["MntFishProducts"]+df["MntSweetProducts"]+df["MntGoldProds"]

df["Total_Spending"]=df["MntWines"]+df["MntFruits"]+df["MntMeatProducts"]+df["MntFishProducts"]+df["MntSweetProducts"]+df["MntGoldProds"]
#educatin
df["Education"].value_counts()
#under graduate,post,graduate
df["Education"]=df["Education"].replace(
    {
        "Basic":"Undergraduate","2n Cycle":"Undergraduate",
        "Graduation":"Graduate",
        "Master":"Postgraduate","PhD":"Postgraduate"
    }
)

df["Education"].value_counts()

#marital status
df["Education"].value_counts()
df["Living_with"]=df["Marital_Status"].replace(
    {
        "Marrid":"partner","Together":"Partner",
        "Single":"Alone",
        "Widow":"Alone","Absurd":"Alone","YOLO":"Alone"
    }
)

df["Marital_Status"].value_counts()

df["Living_with"].value_counts()

#drop unnessary
cols=["ID","Year_Birth","Marital_Status","Kidhome","Teenhome","Dt_Customer"]
spending_col=["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]

cols_to_drop=cols+spending_col

df_cleaned=df.drop(columns=cols_to_drop)

df_cleaned.shape
df_cleaned.head()


#one hot incoding
cols=["Income","Recency","Response","age","Total_Spending","Total_children"]

#related plots of some features 
sns.pairplot(df_cleaned[cols])

#remove outliers
print("data size with outliers",len(df_cleaned))

df_cleaned=df_cleaned[(df_cleaned["age"]<90)]
df_cleaned=df_cleaned[(df_cleaned["Income"]<600_000)]

print("data size without outliers :",len(df_cleaned))

#heatmap
corr=df_cleaned.corr(numeric_only=True)

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)
