# DIFFERENT OPERATIONS ON DATA I WILL PERFORM TO MAKE THE DATA AS CLEAN AS I CAN 
# TO TO I BECOME EASY TO UNDERSTAND IT

import pandas as pd

df = pd.read_csv("Data/raw/ai4i2020.csv")

#MAKING COPY OF OUR DATA SO THE ORIGINAL DATA SHOULD REMAIN SAME
#ANY CHANGE WE MADE IS IN COPY DATA THEN AFTER IT WE SAVE THE COPY DATA
copy_data = df.copy()

# DURING VALIDATION OF DATA WE FIND COLUMN NAMES ARE NOT GOOD 
#SO WE RENAME SOME OF THE COLUMN NAMES

copy_data.rename(columns = {"Rotational speed [rpm]":"Rotational_speed_rpm"})

copy_data.rename(columns = {"TWF":"Tool_Wear_Failure","HDF":"Heat_Dissipation_Failure"}, inplace=True)
copy_data.rename(columns ={"Product ID":"Machine_Id","Type":"Machine_type","Air temperature [K]":"Air_Temp_K"},inplace = True)
copy_data.rename(columns = {"PWF":"Power_Failure","OSF"	:"Overstrain_Failure","RNF":"Random_Failure","Machine failure":"Machine_Failure"}, inplace=True)

copy_data.rename(columns = {"Process temperature [K]":"Process_Temp_K","Rotational speed [rpm]"	:" Rotational_Speed_RPM","Torque [Nm]":"Torque_Nm", "Tool wear [min]":"Tool_Wear_Min"}, inplace=True)
print(copy_data.info())

# to check how many unique machine id (which id not repeate)
# print(copy_data["Machine Id"].nunique())

# print(df["Type"].unique())


copy_data.to_csv("Data/processed/cleaned_data.csv")