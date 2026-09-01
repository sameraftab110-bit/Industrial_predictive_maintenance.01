import pandas as  pd 

df = pd.read_csv("Data/processed/cleaned_data.csv")
#print(df.head())

#adding some new features in the dataeset for help in analyze the data to solve problem

df["temperature_diff"] = df["Process_Temp_K"] - df["Air_Temp_K"]

df["Total_failure_count"] = df["Tool_Wear_Failure"] + df["Heat_Dissipation_Failure"]+df["Power_Failure"] + df["Overstrain_Failure"]+df["Random_Failure"]

df["Torque_Category"] = df["Torque_Nm"].apply(
    lambda x: "Low" if x <= 25
    else "Medium" if   26 <= x <= 45
    else "High"
)


df["RPM_type"] = df[" Rotational_Speed_RPM"].apply(
    lambda x: "Low" if x <= 1250
    else "Medium" if   1251 <= x <= 1650
    else "High"
)


df["Tool_Wear_type"] = df["Tool_Wear_Min"].apply(
    lambda x: "Low" if x <= 80
    else "Medium" if   81 <= x <= 130
    else "High"
)



df["High_Risk_flag"] = df.apply(

    lambda x: "YES" 
                if x["Total_failure_count"] == 1 and
                       x["Torque_Category"]     == "High" and 
                       x["RPM_type"]            == "High" and  
                       x["Tool_Wear_type"]      == "High"  

                else "No" ,

    axis = 1
    )
print(df.describe())

df.to_csv("Data/processed/featured_data.csv")

df = pd.read_csv("Data/processed/featured_data.csv")
print('rrr')
print(df.head())
