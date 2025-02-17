import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#cleansing cpi data
prev_cpi = pd.read_excel("data/raw/cpi_data.csv")
prev_cpi = prev_cpi.drop(columns=["HALF1", "HALF2"])
prev_cpi = prev_cpi.melt(id_vars=["Year"], var_name="Month", value_name="CPI")
prev_cpi["Month"] = prev_cpi["Month"].str.strip().str.capitalize()
month_mapping = {
    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
    "May": "May", "Jun": "June", "Jul": "July", "Aug": "August",
    "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December"
}
prev_cpi["Month"] = prev_cpi["Month"].replace(month_mapping)
prev_cpi["Year"] = prev_cpi["Year"].astype(int)
prev_cpi = prev_cpi.groupby(["Year", "Month"])["CPI"].mean().reset_index()
print(prev_cpi.info())

fed_funds_rate = pd.read_csv("data/raw/DFF.csv")
fed_funds_rate["observation_date"] = pd.to_datetime(fed_funds_rate["observation_date"], errors="coerce")

fed_funds_rate["Year"] = fed_funds_rate["observation_date"].dt.year
fed_funds_rate["Month"] = fed_funds_rate["observation_date"].dt.month

fed_funds_rate.head()
monthly_avg_fed_funds = fed_funds_rate.groupby(["Year", "Month"])["DFF"].mean().reset_index()
monthly_avg_fed_funds.rename(columns={"DFF": "Avg_Fed_Funds_Rate"}, inplace=True)

monthly_avg_fed_funds["Month"] = monthly_avg_fed_funds["Month"].apply(lambda x: pd.to_datetime(str(x), format="%m").strftime("%B"))


print(monthly_avg_fed_funds.head())

gdp = pd.read_csv("data/raw/GDP.csv")

gdp["observation_date"] = pd.to_datetime(gdp["observation_date"], errors="coerce")

gdp["Year"] = gdp["observation_date"].dt.year
gdp["Month"] = gdp["observation_date"].dt.month

gdp.drop(columns = "observation_date")

gdp = gdp.groupby(["Year", "Month"])["GDP"].mean().reset_index()

gdp["Month"] = gdp["Month"].apply(lambda x: pd.to_datetime(str(x), format="%m").strftime("%B"))

print(gdp.head())

ur = pd.read_csv("data/raw/UR.csv")

ur["observation_date"] = pd.to_datetime(ur["observation_date"], errors="coerce")

ur["Year"] = ur["observation_date"].dt.year
ur["Month"] = ur["observation_date"].dt.month

ur.drop(columns = "observation_date")

ur = ur.groupby(["Year", "Month"])["UNRATE"].mean().reset_index()

ur["Month"] = ur["Month"].apply(lambda x: pd.to_datetime(str(x), format="%m").strftime("%B"))

print(ur.head())

M2 = pd.read_csv("data/raw/WM2NS.csv")

M2["observation_date"] = pd.to_datetime(M2["observation_date"], errors="coerce")

M2["Year"] = M2["observation_date"].dt.year
M2["Month"] = M2["observation_date"].dt.month

M2.drop(columns = "observation_date")

M2 = M2.groupby(["Year", "Month"])["WM2NS"].mean().reset_index()

M2["Month"] = M2["Month"].apply(lambda x: pd.to_datetime(str(x), format="%m").strftime("%B"))

print(M2.head())

ppi = pd.read_csv("data/raw/PPIACO.csv")

ppi["observation_date"] = pd.to_datetime(ppi["observation_date"], errors="coerce")

ppi["Year"] = ppi["observation_date"].dt.year
ppi["Month"] = ppi["observation_date"].dt.month

ppi.drop(columns = "observation_date")

ppi = ppi.groupby(["Year", "Month"])["PPIACO"].mean().reset_index()

ppi["Month"] = ppi["Month"].apply(lambda x: pd.to_datetime(str(x), format="%m").strftime("%B"))

print(ppi.info())

ICS = pd.read_excel("data/raw/redbk01a.csv")

ICS.columns = ["Month", "Year", "ICS", "Cases"]

ICS = ICS.dropna(subset=["Year", "Month", "ICS"])

ICS["Year"] = ICS["Year"].astype(int)
ICS["Month"] = ICS["Month"].str.strip().str.capitalize()

ICS = ICS.groupby(["Year", "Month"])["ICS"].mean().reset_index()

print(ICS.head())

vix = pd.read_csv("data/raw/VIXCLS.csv")

vix["observation_date"] = pd.to_datetime(vix["observation_date"], errors="coerce")

vix["Year"] = vix["observation_date"].dt.year
vix["Month"] = vix["observation_date"].dt.month

vix.drop(columns = "observation_date")

vix = vix.groupby(["Year", "Month"])["VIXCLS"].mean().reset_index()

vix["Month"] = vix["Month"].apply(lambda x: pd.to_datetime(str(x), format="%m").strftime("%B"))

print(vix.info())

merged_data = prev_cpi.merge(ppi, on=["Year", "Month"], how="outer")
merged_data = merged_data.merge(monthly_avg_fed_funds, on=["Year", "Month"], how="outer")
merged_data = merged_data.merge(gdp, on=["Year", "Month"], how="outer")
merged_data = merged_data.merge(ur, on=["Year", "Month"], how="outer")
merged_data = merged_data.merge(M2, on=["Year", "Month"], how="outer")
merged_data = merged_data.merge(ICS, on=["Year", "Month"], how="outer")
merged_data = merged_data.merge(vix, on=["Year", "Month"], how="outer")

merged_data.rename(columns={
    "CPI": "CPI_Index",
    "PPIACO": "PPI_Index",
    "Fed_Funds_Rate": "Fed_Funds_Rate",
    "GDP": "GDP_Value",
    "UNRATE": "Unemployment_Rate",
    "WM2NS": "Money_Supply_M2",
    "ICS": "Consumer_Sentiment_Index",
    "VIXCLS": "VIX_Index"
}, inplace=True)

month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

merged_data["Month"] = pd.Categorical(merged_data["Month"], categories=month_order, ordered=True)

merged_data = merged_data.sort_values(by=["Year", "Month"])

merged_data.fillna(method="ffill", inplace=True)
merged_data = merged_data.dropna()
print(merged_data.info)
merged_data.to_csv("data/processed/merged_macro_data.csv", index=False)

file_path = "data/processed/merged_macro_data.csv"  
merged_data = pd.read_csv(file_path)

merged_data["Month_Num"] = pd.to_datetime(merged_data["Month"], format="%B").dt.month

merged_data["Year"] = merged_data["Year"].astype(int)

lag_features = ["CPI_Index", "PPI_Index", "Avg_Fed_Funds_Rate", "GDP_Value", 
                "Unemployment_Rate", "Money_Supply_M2", "Consumer_Sentiment_Index", "VIX_Index"]

for col in lag_features:
    for lag in [1, 3, 6, 12]:  
        merged_data[f"{col}_Lag_{lag}"] = merged_data[col].shift(lag)

for col in lag_features:
    for window in [3, 6, 12]:  
        merged_data[f"{col}_Roll_{window}"] = merged_data[col].rolling(window=window).mean()
        
merged_data["Inflation_Rate"] = merged_data["CPI_Index"].pct_change() * 100

merged_data["PPI_CPI_Ratio"] = merged_data["PPI_Index"] / merged_data["CPI_Index"]
merged_data["Fed_Unemp_Ratio"] = merged_data["Avg_Fed_Funds_Rate"] / merged_data["Unemployment_Rate"]
merged_data["Money_GDP_Ratio"] = merged_data["Money_Supply_M2"] / merged_data["GDP_Value"]

volatility_features = ["CPI_Index", "PPI_Index", "VIX_Index"]

for col in volatility_features:
    merged_data[f"{col}_Volatility"] = merged_data[col].rolling(window=6).std()

merged_data["Fed_Unemp_Interaction"] = merged_data["Avg_Fed_Funds_Rate"] * merged_data["Unemployment_Rate"]
merged_data["CPI_Money_Interaction"] = merged_data["CPI_Index"] * merged_data["Money_Supply_M2"]

merged_data["PPI_Lead_3"] = merged_data["PPI_Index"].shift(-3)
merged_data["Fed_Rate_Lead_6"] = merged_data["Avg_Fed_Funds_Rate"].shift(-6)

merged_data.fillna(method="ffill", inplace=True) 

merged_data.dropna(inplace=True)

processed_file_path = "data/processed/feature_engineered_macro_data.csv"
merged_data.to_csv(processed_file_path, index=False)


numeric_data = merged_data.select_dtypes(include=["number"])

cpi_correlation = numeric_data.corr()["CPI_Index"].sort_values(ascending=False)

print("Top Features Most Positively Correlated with CPI Index:**")
print(cpi_correlation.head(10))

print("Top Features Most Negatively Correlated with CPI Index:")
print(cpi_correlation.tail(10))

numeric_data = merged_data.select_dtypes(include=["number"])

cpi_correlation = numeric_data.corr()["CPI_Index"].sort_values(ascending=False)

num_features = len(cpi_correlation) - 1 
fig, axes = plt.subplots(nrows=num_features, figsize=(8, num_features * 3))

for i, (feature, correlation) in enumerate(cpi_correlation.items()):
    if feature != "CPI_Index":  
        ax = axes[i - 1]
        sns.regplot(x=numeric_data[feature], y=numeric_data["CPI_Index"], ax=ax, scatter_kws={"s": 10}, line_kws={"color": "red"})
        ax.set_title(f"Correlation with CPI_Index: {feature} ({correlation:.2f})", fontsize=12)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("CPI_Index", fontsize=10)
plt.tight_layout()
plt.show()