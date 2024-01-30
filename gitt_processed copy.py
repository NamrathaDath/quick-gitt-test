import valibrary as vl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.optimize import curve_fit

# Import the parameter file
ocv_path = "positive_ocv.csv"
U_df = pd.read_csv(ocv_path)

#scipy.interpolate.interp1d is used to perform linear interpolation on the OCV data.
U = scipy.interpolate.interp1d(
    U_df["Stoichiometry"], U_df["Positive electrode OCP [V]"], kind="linear"
)
U_inv = scipy.interpolate.interp1d(
    U_df["Positive electrode OCP [V]"], U_df["Stoichiometry"], kind="linear"
)

path = "230704_GPA-A_B0_GITT_Dch_(T25_NV008)_C1_C01.txt"
converter = vl.BiologicConverter(path)

df = converter.to_cloud()

actual_df = converter.to_cloud()

idx = df["current"] < 0 #index for when current from the dataframe is laess than 0
discharge_df = df[idx]  #new dataframe which includes only negative values of current from the old dataframe

discharge_step_numbers = discharge_df["step_number"].unique() #another dataframe where data is filtered based on unique step numbers

initial_voltage = []
instantaneous_drop_voltage = []
final_voltage = []
final_voltage_last_point = []
instantaneous_rise_voltage = []

initial_time = []
instantaneous_drop_time = []
final_time = []
instantaneous_rise_time = []
time_pulse = []
start_time = []

initial_current = []
final_current = []

# Lists to store internal resistance values
internal_resistance_drop = []
internal_resistance_rise = []

# Lists to store diffusion coefficients
diffusion_coefficients = []

# Function representing Sand's equation
def sand_equation(t, OCV0, a):
    return OCV0 - a * np.sqrt(t)


# Function to fit Sand's equation and extract diffusion coefficients
def fit_sands_equation(t, OCV):

    # curve_fit is a function from scipy.optimize that fits a function to data
    popt, _ = curve_fit(sand_equation, t, OCV)

    # popt contains the optimized parameters of the fitted function
    OCV0_fit, a_fit = popt

    # The diffusion coefficient is calculated from the fitted 'a' parameter
    diffusion_coefficient = a_fit**2 / 4
    return OCV0_fit, diffusion_coefficient


for i in discharge_step_numbers:
    Before_pulse = df[df["step_number"] == i-1]  #extracting a subset of the DataFrame 'df' where the step no. colum has a specific value (i-1)
    pulse = discharge_df[discharge_df["step_number"] == i]
    After_pulse = df[df["step_number"] == i+1]
    
    initial_voltage.append(Before_pulse["voltage"].iloc[-1])
    instantaneous_drop_voltage.append(pulse["voltage"].iloc[1])
    final_voltage.append(pulse["voltage"].iloc[-1]) 
    final_voltage_last_point.append(After_pulse["voltage"].iloc[-1])
    instantaneous_rise_voltage.append(After_pulse["voltage"].iloc[0])

    # initial_time.append(pulse["total_time_millis"].iloc[0])
    initial_time.append(Before_pulse["total_time_millis"].iloc[-1])
    instantaneous_drop_time.append(pulse["total_time_millis"].iloc[1])

    start_time.append(pulse_df["total_time_millis"].iloc[0])
    pulse_df["pulse_time_ms"] = pulse_df["total_time_millis"] - t0
    df["pulse_time_ms"] = df["total_time_millis"] - t0

    final_time.append(pulse["total_time_millis"].iloc[-1])
    instantaneous_rise_time.append(After_pulse["total_time_millis"].iloc[0])

    v_pulse = scipy.interpolate.interp1d(
    pulse_df["pulse_time_ms"], pulse_df["voltage"]
)

    initial_current.append(pulse["current"].iloc[1])
    final_current.append(pulse["current"].iloc[-1])

   '''  # Fit Sand's equation and extract diffusion coefficient
    t_relaxation = actual_df["total_time_millis"][
        (actual_df["step_number"] == i)
        & (actual_df["current"] < 0)
    ].values'''

 '''   OCV_relaxation = actual_df["voltage"][
        (actual_df["step_number"] == i)
        & (actual_df["current"] < 0)
    ].values'''

    '''OCV0_fit, diffusion_coefficient = fit_sands_equation(t_relaxation, OCV_relaxation)
    
    # Append the diffusion coefficient to the list
    diffusion_coefficients.append(diffusion_coefficient)

    print(f"Step number {i}: Diffusion Coefficient = {diffusion_coefficient}")'''

initial_stochiometry = U_inv(initial_voltage)
final_stochiometry = U_inv(final_voltage_last_point )


# Calculate voltage drop and rise
voltage_drop = np.array(instantaneous_drop_voltage) - np.array(initial_voltage)
voltage_rise = np.array(final_voltage) - np.array(instantaneous_rise_voltage)

time_pulse = (np.array(final_time) - np.array(instantaneous_drop_time)) * 1e-3



#t_pulse =np.array(instantaneous_drop_time) -np.array

# Calculate average current as (initial current + final current) / 2
average_current = (np.array(initial_current) + np.array(final_current)) / 2

internal_resistance_drop = np.array(voltage_drop)/np.array(average_current)
internal_resistance_rise = np.array(voltage_rise)/np.array(average_current)


# Print or use the calculated values as needed
for i, step_number in enumerate(discharge_step_numbers):
    print(f"Step number {step_number}: Voltage Drop = {voltage_drop[i]}, Voltage Rise = {voltage_rise[i]},"
          f"Average Current = {average_current[i]}"
          f"Resistance Drop = {internal_resistance_drop[i]}, Resistance Rise = {internal_resistance_rise[i]} , Pulse Time = {time_pulse[i]}")


fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(actual_df["total_time_millis"], actual_df["current"])
ax[0].scatter(instantaneous_drop_time, initial_current, color='blue', marker='o')
ax[0].scatter(final_time, final_current, color='orange', marker='o')
ax[0].set_ylabel('Current (A)')


ax[1].plot(actual_df["total_time_millis"], actual_df["voltage"])
ax[1].scatter(initial_time, initial_voltage, color='red', marker='o')
ax[1].scatter(instantaneous_drop_time, instantaneous_drop_voltage, color='blue', marker='o')
ax[1].scatter(final_time, final_voltage, color='orange', marker='o')
ax[1].scatter(instantaneous_rise_time, instantaneous_rise_voltage, color='black', marker='o')

ax[1].set_ylabel('Voltage (V)')



# Function to fit Sand's equation and extract diffusion coefficients
def fit_sands_equation(t, OCV):
    popt, _ = curve_fit(sand_equation, t, OCV)
    OCV0_fit, a_fit = popt
    diffusion_coefficient = a_fit**2 / 4
    return OCV0_fit, diffusion_coefficient




# Create a dataframe to save it to excel
# processed_data = pd.DataFrame()
# processed_data["initial_voltage"] = initial_voltage
# processed_data["initial_time"] = initial_time
# processed_data["instant_drop_voltage"] = instantaneous_drop_voltage
# processed_data["instant_drop_time"] = instantaneous_drop_time


# processed_data.to_csv("Processed GITT.csv", index=False)


plt.show()
print("Hi")




