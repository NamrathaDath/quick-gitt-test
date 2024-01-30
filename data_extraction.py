import valibrary as vl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.optimize import curve_fit
import scipy.interpolate

dc_steps = [3, 47]

# Import the parameter file
ocv_path = "GPA-A_B1_C2_pOCV_xLi_Average.csv"
U_df = pd.read_csv(ocv_path)

#Linear interpolation is performed on the OCV data to create functions U and U_inv. These functions interpolate voltage based on stoichiometry and vice versa.
#scipy.interpolate.interp1d is used to perform linear interpolation on the OCV data.
U = scipy.interpolate.interp1d(
    U_df["Stoichiometry"], U_df["Positive electrode OCP [V]"], kind="linear", fill_value="extrapolate"
)
U_inv = scipy.interpolate.interp1d(
    U_df["Positive electrode OCP [V]"], U_df["Stoichiometry"], kind="linear", fill_value="extrapolate"
)

# GITT data
path = "231213_GPA-A_B1_GITT_(T25_NV008)_C2_C06.txt"
converter = vl.BiologicConverter(path)
df = converter.to_cloud()

fig, ax = plt.subplots(5, 1, figsize=(16, 9), sharex=True)

ax[0].plot(df["total_time_millis"], df["current"])
# ax[1].plot(df["total_time_millis"], df["voltage_cathode"])
ax[2].plot(df["total_time_millis"], df["voltage"])
ax[3].plot(df["total_time_millis"], df["(Q-Qo)/mA.h"])
ax[4].plot(df["total_time_millis"], df["step_number"])

ax[0].set_ylabel("Current (mA)")
# ax[1].set_ylabel("Voltage_Cathode (V)")
ax[2].set_ylabel("Voltage (V)")
ax[3].set_ylabel("Capacity (mA.h)")
ax[4].set_ylabel("Step number")

ax[-1].set_xlabel("Time (ms)")

for a in ax: 
    a.grid(True)


# 
idx = np.logical_and(df["step_number"] >= dc_steps[0], df["step_number"] <= dc_steps[1])
dc_steps = df[idx]

step_df = dc_steps.groupby("step_number").agg(
    capacity_ah=("(Q-Qo)/mA.h", "last"),
    voltage=("voltage", "last"),
    step_type=("step_type", "first"),
)

# only keep the rest steps
step_df = step_df[step_df["step_type"] == "REST"]

fig, ax = plt.subplots(1, 1, figsize=(16, 9))

gitt_v_interp = scipy.interpolate.interp1d(
    step_df["capacity_ah"], step_df["voltage"], kind="linear", fill_value="extrapolate"
)
gitt_v_inv = scipy.interpolate.interp1d(
    step_df["voltage"], step_df["capacity_ah"], kind="linear", fill_value="extrapolate"
)

v1 = step_df["voltage"].iloc[3]
v2 = step_df["voltage"].iloc[-8]

sto1 = U_inv(v1)
sto2 = U_inv(v2)

capa1 = gitt_v_inv(v1)
capa2 = gitt_v_inv(v2)

# scale the capacity onto the stoichiometry 
sto = (sto2 - sto1) / (capa2 - capa1) * (step_df["capacity_ah"] - capa1) + sto1
step_df["stoichiometry"] = sto
step_df.drop_duplicates(subset='stoichiometry', inplace=True)
sto = step_df["stoichiometry"]
gitt_sto_v_interp = scipy.interpolate.interp1d(
    sto, step_df["voltage"], kind="cubic", fill_value="extrapolate"
)

ax.plot(sto, step_df["voltage"], "o", label="GITT Points")
ax.plot(U_df["Stoichiometry"], gitt_sto_v_interp(U_df["Stoichiometry"]), "-", label="GITT (Interp)")
ax.plot(U_df["Stoichiometry"], U_df["Positive electrode OCP [V]"], "-", label="pOCV")

ax.plot(sto1, v1, "x", label="V1", color="r")
ax.plot(sto2, v2, "+", label="V2", color="g")

ax.set_xlabel("Capacity (mA.h)")
ax.set_ylabel("Voltage (V)")
ax.grid(True)
ax.legend()

plt.show()

print("")
