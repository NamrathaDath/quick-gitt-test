import valibrary as vl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.optimize import curve_fit
import scipy.interpolate

# Import the c_tilde solution
c_tilde_df = pd.read_csv("c_tilde_surf.csv")
c_tilde = scipy.interpolate.interp1d(
    c_tilde_df["t_tilde"], c_tilde_df["c_tilde_surf"], kind="linear", fill_value="extrapolate"
)

# Import the parameter file
fit_Rp = True
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


path = "231030_GPA-A-B1_GITT_(T25_NV008)_A3122_C12.txt"
converter = vl.BiologicConverter(path)

df = converter.to_cloud()

#actual_df = converter.to_cloud()
converter = vl.BiologicConverter(path)

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

tpulse = []

initial_current = []
final_current = []

before_discharge_pulses = []
discharge_pulses = []

# Lists to store internal resistance values
internal_resistance_drop = []
internal_resistance_rise = []

# Lists to store diffusion coefficients
diffusion_coefficients = []

for i in discharge_step_numbers:
    Before_pulse = df[df["step_number"] == i-1]  #extracting a subset of the DataFrame 'df' where the step no. colum has a specific value (i-1)
    pulse = discharge_df[discharge_df["step_number"] == i]
    After_pulse = df[df["step_number"] == i+1]
    
    initial_voltage.append(Before_pulse["voltage_cathode"].iloc[-1])
    instantaneous_drop_voltage.append(pulse["voltage_cathode"].iloc[1])
    final_voltage.append(pulse["voltage_cathode"].iloc[-1]) 
    final_voltage_last_point.append(After_pulse["voltage_cathode"].iloc[-1])
    instantaneous_rise_voltage.append(After_pulse["voltage_cathode"].iloc[0])

    # initial_time.append(pulse["total_time_millis"].iloc[0])
    initial_time.append(Before_pulse["total_time_millis"].iloc[-1])
    instantaneous_drop_time.append(pulse["total_time_millis"].iloc[1])

    pulse_start_time = pulse["total_time_millis"].iloc[0]

    pulse["processed_time_s"] = (pulse["total_time_millis"] - pulse["total_time_millis"].iloc[0]) * 1e-3
    tpulse.append(pulse["processed_time_s"].max())

    final_time.append(pulse["total_time_millis"].iloc[-1])
    instantaneous_rise_time.append(After_pulse["total_time_millis"].iloc[0])

    initial_current.append(pulse["current"].iloc[1])
    final_current.append(pulse["current"].iloc[-1])

    Before_pulse["processed_time_s"] = (Before_pulse["total_time_millis"] - pulse_start_time) * 1e-3
    before_discharge_pulses.append(Before_pulse)
    discharge_pulses.append(pulse)

   
initial_stochiometry = U_inv(initial_voltage)
final_stochiometry = U_inv(final_voltage_last_point )


# Calculate voltage drop and rise
v_drop = np.array(instantaneous_drop_voltage) - np.array(initial_voltage)
v_rise = np.array(final_voltage) - np.array(instantaneous_rise_voltage)

time_pulse = (np.array(final_time) - np.array(instantaneous_drop_time)) * 1e-3



#t_pulse =np.array(instantaneous_drop_time) -np.array

# Calculate average current as (initial current + final current) / 2
average_current = (np.array(initial_current) + np.array(final_current)) / 2

internal_resistance_drop = np.array(v_drop)/np.array(average_current)
internal_resistance_rise = np.array(v_rise)/np.array(average_current)


# Print or use the calculated values as needed
for i, step_number in enumerate(discharge_step_numbers):
    print(f"Step number {step_number}: Voltage Drop = {v_drop[i]}, Voltage Rise = {v_rise[i]},"
          f"Average Current = {average_current[i]}"
          f"Resistance Drop = {internal_resistance_drop[i]}, Resistance Rise = {internal_resistance_rise[i]} , Pulse Time = {time_pulse[i]}")


fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(actual_df["total_time_millis"], actual_df["current"])
ax[0].scatter(instantaneous_drop_time, initial_current, color='blue', marker='o')
ax[0].scatter(final_time, final_current, color='orange', marker='o')
ax[0].set_ylabel('Current (A)')


ax[1].plot(actual_df["total_time_millis"], actual_df["voltage_cathode"])
ax[1].scatter(initial_time, initial_voltage, color='red', marker='o')
ax[1].scatter(instantaneous_drop_time, instantaneous_drop_voltage, color='blue', marker='o')
ax[1].scatter(final_time, final_voltage, color='orange', marker='o')
ax[1].scatter(instantaneous_rise_time, instantaneous_rise_voltage, color='black', marker='o')

ax[1].set_ylabel('Voltage (V)')

t_est = []
Rp_est = []

# i = 15
# tpulse = tpulse[i]
# initial_stochiometry = initial_stochiometry[i]
# final_stochiometry = final_stochiometry[i]
# pulse = discharge_pulses[i]
# average_current = average_current[i]
# v_drop = v_drop[i]

# # Calculate tau and Rp
# def get_voltage(t, tau_D, R):
#     sto_surf = (2 / (3 * np.sqrt(np.pi))) * (final_stochiometry - initial_stochiometry) / tpulse * np.sqrt(
#         t * tau_D
#     ) + initial_stochiometry
#     V = U(sto_surf) + R * average_current
#     return V

# t_pulse_data = np.linspace(1, tpulse, 1000)

# v_pulse = scipy.interpolate.interp1d(
#     pulse["processed_time_s"], pulse["voltage"]
#     )

# voltage_data = v_pulse(t_pulse_data)

# # curve fit the vp data
# R_est = v_drop / average_current
# tau_D_est = 3600


# if fit_Rp:
#     popt_p, _ = scipy.optimize.curve_fit(
#     get_voltage, t_pulse_data, voltage_data, p0=[tau_D_est, R_est], bounds=(0, np.inf)
#     )
# else:
#     popt_p, _ = scipy.optimize.curve_fit(
#     lambda t, tau_D: get_voltage(t, tau_D, R_est), t_pulse_data, voltage_data, p0=[tau_D_est], bounds=(0, np.inf)
#     )
#     popt_p = [popt_p[0], R_est]

# print(f"Initial tau_Dp: {tau_D_est}, fitted tau_Dp: {popt_p[0]}")
# print(f"Initial Rp: {R_est}, fitted Rp: {popt_p[1]}")

# fig, ax = plt.subplots(1, 1, sharex=True, figsize=(9, 6))

# voltage_sim = get_voltage(t_pulse_data, popt_p[0], popt_p[1])

# ax.plot(pulse["processed_time_s"], pulse["voltage"], label="Data")
# ax.plot(t_pulse_data, voltage_sim, linestyle="--", label="Fit")

# ax.set_ylabel("Potential [V]")

# ax.legend()
# ax.set_xlabel("Time [s]")

# ax.grid(True)

for i in range(len(tpulse)):

    # Calculate tau and Rp
    def get_voltage_sand(t, tau_D, R):
        sto_surf = (2 / (3 * np.sqrt(np.pi))) * (final_stochiometry[i] - initial_stochiometry[i]) / tpulse[i] * np.sqrt(
            t * tau_D
        ) + initial_stochiometry[i]
        V = U(sto_surf) + R * average_current[i]
        return V
    
    def get_voltage(t, tau_D, R):
        t_tilde = t / tau_D
        sto_surf = 1/3 * (final_stochiometry[i] - initial_stochiometry[i]) * tau_D / tpulse[i] * c_tilde(t_tilde) + initial_stochiometry[i]
        V = U(sto_surf) + R * average_current[i]
        return V

    t_pulse_data = np.linspace(1, tpulse[i], 1000)

    pulse = discharge_pulses[i]
    v_pulse = scipy.interpolate.interp1d(
        pulse["processed_time_s"], pulse["voltage_cathode"]
        )

    voltage_data = v_pulse(t_pulse_data)

    # curve fit the vp data
    R_est = v_drop[i] / average_current[i]
    tau_D_est = 3600


    if fit_Rp:
        popt_p, _ = scipy.optimize.curve_fit(
        get_voltage, t_pulse_data, voltage_data, p0=[tau_D_est, R_est], bounds=(0, np.inf)
        )
    else:
        popt_p, _ = scipy.optimize.curve_fit(
        lambda t, tau_D: get_voltage(t, tau_D, R_est), t_pulse_data, voltage_data, p0=[tau_D_est], bounds=(0, np.inf)
        )
        popt_p = [popt_p[0], R_est]

    print(f"Initial tau_Dp: {tau_D_est}, fitted tau_Dp: {popt_p[0]}")
    print(f"Initial Rp: {R_est}, fitted Rp: {popt_p[1]}")

    t_est.append(popt_p[0])
    Rp_est.append(popt_p[1])

def get_sim_voltage_sands(t, tau_D, R, final_stochiometry, initial_stochiometry, average_current, tpulse):
    sto_surf = (2 / (3 * np.sqrt(np.pi))) * (final_stochiometry - initial_stochiometry) / tpulse * np.sqrt(
            t * tau_D
        ) + initial_stochiometry
    V = U(sto_surf) + R * average_current
    return V

def get_sim_voltage(t, tau_D, R, final_stochiometry, initial_stochiometry, average_current, tpulse):
    t_tilde = t / tau_D
    sto_surf = 1/3 * (final_stochiometry - initial_stochiometry) * tau_D / tpulse * c_tilde(t_tilde) + initial_stochiometry
    V = U(sto_surf) + R * average_current
    return V

def get_sim_ocv(t, tau_D, R, final_stochiometry, initial_stochiometry, average_current, tpulse):
    sto_av = initial_stochiometry + (final_stochiometry - initial_stochiometry) / tpulse * t
    return U(sto_av)

fig, ax = plt.subplots(5, 4, sharex=True, figsize=(3, 2))
ax = ax.flatten()
for i in range(20):
    t_pulse_data = np.linspace(1, tpulse[i], 1000)

    voltage_sim = get_sim_voltage(t_pulse_data, t_est[i], Rp_est[i], final_stochiometry[i], 
                                  initial_stochiometry[i], average_current[i], tpulse[i])
    
    ocv_sim = get_sim_ocv(t_pulse_data, t_est[i], Rp_est[i], final_stochiometry[i], 
                                  initial_stochiometry[i], average_current[i], tpulse[i])
    pulse = discharge_pulses[i]
    before_pulse = before_discharge_pulses[i]

    ax[i].plot(pulse["processed_time_s"], pulse["voltage_cathode"], label="Data")
    ax[i].plot(before_pulse["processed_time_s"], before_pulse["voltage_cathode"], color="k")
    ax[i].plot(t_pulse_data, voltage_sim, linestyle="--", label="Fit")
    ax[i].plot(t_pulse_data, ocv_sim, linestyle=":", label="OCV")

    # ax[i].set_ylabel("Potential [V]")

    # ax[i].legend()
    # ax[i].set_xlabel("Time [s]")

    ax[i].set_xlim([-30, max(t_pulse_data)])

    ax[i].grid(True)


plt.show()

'''# Assuming df is your GITT data DataFrame
# For demonstration purposes, you might need to adjust the column names based on your actual DataFrame

# Example GITT data
data = {
    'total_time_millis': [0, 1000, 2000, 3000],
    'current': [0, -2, 1, -3]
}

df = pd.DataFrame(data)

# Calculate total charge passed
total_charge = np.trapz(np.abs(df['current']) / 1000, df['total_time_millis'] / 1000)

print(f'Total Charge Passed: {total_charge} Coulombs')'''


print("Hi")




