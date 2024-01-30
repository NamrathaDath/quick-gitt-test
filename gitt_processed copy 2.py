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

#data to be imported
path = "231213_GPA-A_B1_GITT_(T25_NV008)_C2_C06.txt"
converter = vl.BiologicConverter(path)

df = converter.to_cloud()

converter = vl.BiologicConverter(path)

idx = df["current"] < -1e-4 #index for when current from the dataframe is less than 0
idx_charge = df["current"] > 2e-4 #index for when current from the dataframe is greater than 0

discharge_df = df[idx]  #new dataframe which includes only negative values of current from the old dataframe
charge_df = df[idx_charge] #new dataframe which includes only positive values of current from the old dataframe

discharge_step_numbers = discharge_df["step_number"].unique() #another dataframe where data is filtered based on unique step numbers
discharge_step_numbers = discharge_step_numbers[discharge_step_numbers<65]

charge_step_numbers = charge_df["step_number"].unique() #another dataframe where data is filtered based on unique step numbers

initial_voltage = []
intial_voltage_charge = []

instantaneous_drop_voltage = []
instantaneous_drop_voltage_charge = []

instantaneous_rise_voltage = []
instantaneous_rise_voltage_charge = []

final_voltage = []
final_voltage_charge = []

final_voltage_last_point = []
final_voltage_last_point_charge = []

initial_time = []
instantaneous_drop_time = []
final_time = []
instantaneous_rise_time = []
start_time = []

tpulse_discharge = []
tpulse_charge = []

initial_time_charge = []
instantaneous_drop_time_charge = []
final_time_charge = []
instantaneous_rise_time_charge = []

initial_current = []
initial_current_charge = []

final_current = []
final_current_charge = []

before_discharge_pulses = []
discharge_pulses = []

before_charge_pulses = []
charge_pulses = []

# Lists to store internal resistance values
internal_resistance_drop = []
internal_resistance_rise = []

# Lists to store diffusion coefficients
diffusion_coefficients = []

for i in discharge_step_numbers:
    Before_pulse = df[df["step_number"] == i-1]  #extracting a subset of the DataFrame 'df' where the step no. colum has a specific value (i-1)
    pulse = discharge_df[discharge_df["step_number"] == i] #extracting a subset of the DataFrame 'df' where the step no. colum has a specific value (i)
    After_pulse = df[df["step_number"] == i+1] #extracting a subset of the DataFrame 'df' where the step no. colum has a specific value (i+1)
    
    initial_voltage.append(Before_pulse["voltage"].iloc[-1])
    instantaneous_drop_voltage.append(pulse["voltage"].iloc[1])
    final_voltage.append(pulse["voltage"].iloc[-1]) 
    final_voltage_last_point.append(After_pulse["voltage"].iloc[-1])
    instantaneous_rise_voltage.append(After_pulse["voltage"].iloc[0])

    # initial_time.append(pulse["total_time_millis"].iloc[0])
    initial_time.append(Before_pulse["total_time_millis"].iloc[-1])
    instantaneous_drop_time.append(pulse["total_time_millis"].iloc[1])

    pulse_start_time = pulse["total_time_millis"].iloc[0]

    pulse["processed_time_s"] = (pulse["total_time_millis"] - pulse_start_time) * 1e-3
    tpulse_discharge.append(pulse["processed_time_s"].max())

    final_time.append(pulse["total_time_millis"].iloc[-1])
    instantaneous_rise_time.append(After_pulse["total_time_millis"].iloc[0])

    initial_current.append(pulse["current"].iloc[1])
    final_current.append(pulse["current"].iloc[-1])

    Before_pulse["processed_time_s"] = (Before_pulse["total_time_millis"] - pulse_start_time) * 1e-3
    before_discharge_pulses.append(Before_pulse)
    discharge_pulses.append(pulse)


for i in charge_step_numbers:
    Before_pulse_charge = df[df["step_number"] == i-1]  #extracting a subset of the DataFrame 'df' where the step no. colum has a specific value (i-1)
    pulse_charge = charge_df[charge_df["step_number"] == i] #extracting a subset of the DataFrame 'df' where the step no. colum has a specific value (i)
    After_pulse_charge = df[df["step_number"] == i+1] #extracting a subset of the DataFrame 'df' where the step no. colum has a specific value (i+1)
    
    intial_voltage_charge.append(Before_pulse_charge["voltage"].iloc[-1])
    instantaneous_rise_voltage_charge.append(pulse_charge["voltage"].iloc[1])
    final_voltage_charge.append(pulse_charge["voltage"].iloc[-1]) 
    final_voltage_last_point_charge.append(After_pulse_charge["voltage"].iloc[-1])
    instantaneous_drop_voltage_charge.append(After_pulse_charge["voltage"].iloc[1])
    
    initial_time_charge.append(Before_pulse_charge["total_time_millis"].iloc[-1])
    instantaneous_rise_time_charge.append(pulse_charge["total_time_millis"].iloc[1])

    pulse_start_time_charge = pulse_charge["total_time_millis"].iloc[0]

    pulse_charge["processed_time_s"] = (pulse_charge["total_time_millis"] - pulse_start_time_charge) * 1e-3
    tpulse_charge.append(pulse["processed_time_s"].max())

    final_time_charge.append(pulse_charge["total_time_millis"].iloc[-1])
    instantaneous_drop_time_charge.append(After_pulse_charge["total_time_millis"].iloc[0])

    initial_current_charge.append(pulse_charge["current"].iloc[1])
    final_current_charge.append(pulse_charge["current"].iloc[-1])

    Before_pulse_charge["processed_time_s"] = (Before_pulse_charge["total_time_millis"] - pulse_start_time_charge) * 1e-3
    before_charge_pulses.append(Before_pulse_charge)
    charge_pulses.append(pulse_charge)


# Calculate initial and final stochiometry   
initial_stochiometry = U_inv(initial_voltage)
final_stochiometry = U_inv(final_voltage_last_point )

initial_stochiometry_charge = U_inv(intial_voltage_charge)
final_stochiometry_charge = U_inv(final_voltage_last_point_charge)

# Calculate voltage drop and rise
v_drop = np.array(instantaneous_drop_voltage) - np.array(initial_voltage)
v_rise = np.array(final_voltage) - np.array(instantaneous_rise_voltage)

# Calculate voltage drop and rise for charge
v_drop_charge = np.array(instantaneous_rise_voltage_charge) - np.array(intial_voltage_charge)
v_rise_charge = np.array(final_voltage_charge) - np.array(instantaneous_drop_voltage_charge)

# Calculate average current 
average_current = (np.array(initial_current) + np.array(final_current)) / 2
average_current_charge = (np.array(initial_current_charge) + np.array(final_current_charge)) / 2

# Calculate internal resistance
internal_resistance_drop = np.array(v_drop)/np.array(average_current)
internal_resistance_rise = np.array(v_rise)/np.array(average_current)
internal_resistance_drop_charge = np.array(v_drop_charge)/np.array(average_current_charge)
internal_resistance_rise_charge = np.array(v_rise_charge)/np.array(average_current_charge)


# Print or use the calculated values as needed
for i, step_number in enumerate(discharge_step_numbers):
    print(f"Step number {step_number}: Voltage Drop = {v_drop[i]}, Voltage Rise = {v_rise[i]},"
          f"Average Current = {average_current[i]}"
          f"Resistance Drop = {internal_resistance_drop[i]}, Resistance Rise = {internal_resistance_rise[i]} ,"
          f"Pulse Time = {tpulse_discharge[i]}")

for i, step_number in enumerate(charge_step_numbers):
    print(f"Step number {step_number}: Voltage Rise = {v_rise_charge[i]}, Voltage Drop = {v_drop_charge[i]}, "
          f"Average Current = {average_current_charge[i]}"
          f"Resistance Rise = {internal_resistance_rise_charge[i]} , Resistance Drop = {internal_resistance_drop_charge[i]},  Pulse Time = {tpulse_charge[i]}")


fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(df["total_time_millis"], df["current"])
ax[0].scatter(instantaneous_drop_time, initial_current, color='blue', marker='o')
ax[0].scatter(instantaneous_rise_time_charge, initial_current_charge, color='blue', marker='o')
ax[0].scatter(final_time, final_current, color='orange', marker='o')
ax[0].scatter(final_time_charge, final_current_charge, color='orange', marker='o')
ax[0].set_ylabel('Current (A)')

ax[1].plot(df["total_time_millis"], df["voltage"])
ax[1].scatter(initial_time, initial_voltage, color='red', marker='o')
ax[1].scatter(initial_time_charge, intial_voltage_charge, color='red', marker='o')
ax[1].scatter(instantaneous_drop_time, instantaneous_drop_voltage, color='blue', marker='o')
ax[1].scatter(instantaneous_rise_time_charge, instantaneous_rise_voltage_charge, color='blue', marker='o')
ax[1].scatter(final_time, final_voltage, color='orange', marker='o')
ax[1].scatter(final_time_charge, final_voltage_charge, color='orange', marker='o')
ax[1].scatter(instantaneous_rise_time, instantaneous_rise_voltage, color='black', marker='o')
ax[1].scatter(instantaneous_drop_time_charge, instantaneous_drop_voltage_charge, color='black', marker='o')
ax[1].set_ylabel('Voltage (V)')


t_est = []
Rp_est = []


for i in range(len(tpulse_discharge)):

    # Calculate tau and Rp
    def get_voltage_sand(t, tau_D, R):
        sto_surf = (2 / (3 * np.sqrt(np.pi))) * (final_stochiometry[i] - initial_stochiometry[i]) / tpulse_discharge[i] * np.sqrt(
            t * tau_D
        ) + initial_stochiometry[i]
        V = U(sto_surf) + R * average_current[i]
        return V
    
    def get_voltage(t, tau_D, R):
        t_tilde = t / tau_D
        sto_surf = 1/3 * (final_stochiometry[i] - initial_stochiometry[i]) * tau_D / tpulse_discharge[i] * c_tilde(t_tilde) + initial_stochiometry[i]
        V = U(sto_surf) + R * average_current[i]
        return V

    t_pulse_data_discharge = np.linspace(1, tpulse_discharge[i], 1000)

    pulse = discharge_pulses[i]
    v_pulse = scipy.interpolate.interp1d(
        pulse["processed_time_s"], pulse["voltage"]
        )

    voltage_data = v_pulse(t_pulse_data_discharge)

    # curve fit the vp data
    R_est = v_drop[i] / average_current[i]
    R_est_1 = v_rise[i] / average_current[i]
    tau_D_est = 3600


    if fit_Rp:
        popt_p, _ = scipy.optimize.curve_fit(
        get_voltage, t_pulse_data_discharge, voltage_data, p0=[tau_D_est, R_est], bounds=(np.array([0,min(R_est, R_est_1)]), np.array([np.inf, max(R_est, R_est_1)]))
        )
    else:
        popt_p, _ = scipy.optimize.curve_fit(
        lambda t, tau_D: get_voltage(t, tau_D, R_est), t_pulse_data_discharge, voltage_data, p0=[tau_D_est], bounds=(0, np.inf)
        )
        popt_p = [popt_p[0], R_est]

    print(f"Initial tau_Dp: {tau_D_est}, fitted tau_Dp: {popt_p[0]}")
    print(f"Initial Rp: {R_est}, fitted Rp: {popt_p[1]}")

    t_est.append(popt_p[0])
    Rp_est.append(popt_p[1])


t_est_charge = []
Rp_est_charge = []


for i in range(len(tpulse_charge)):

    # Calculate tau and Rp
    def get_voltage_sand(t, tau_D, R):
        sto_surf = (2 / (3 * np.sqrt(np.pi))) * (final_stochiometry_charge[i] - initial_stochiometry_charge[i]) / tpulse_charge[i] * np.sqrt(
            t * tau_D
        ) + initial_stochiometry_charge[i]
        V = U(sto_surf) + R * average_current_charge[i]
        return V
    
    def get_voltage(t, tau_D, R):
        t_tilde = t / tau_D
        sto_surf = 1/3 * (final_stochiometry_charge[i] - initial_stochiometry_charge[i]) * tau_D / tpulse_charge[i] * c_tilde(t_tilde) + initial_stochiometry_charge[i]
        V = U(sto_surf) + R * average_current_charge[i]
        return V

    t_pulse_data_charge = np.linspace(1, tpulse_charge[i], 1000)

    pulse_charge = charge_pulses[i]
    v_pulse_charge = scipy.interpolate.interp1d(
        pulse_charge["processed_time_s"], pulse_charge["voltage"], kind="linear", fill_value="extrapolate"
        )

    voltage_data = v_pulse_charge(t_pulse_data_charge)

    # curve fit the vp data
    R_est = v_drop_charge[i] / average_current_charge[i]
    R_est_1 = v_rise_charge[i] / average_current_charge[i]
    tau_D_est = 3600


    if fit_Rp:
        popt_p, _ = scipy.optimize.curve_fit(
        get_voltage, t_pulse_data_charge, voltage_data, p0=[tau_D_est, R_est], bounds=(np.array([0,min(R_est, R_est_1)]), np.array([np.inf, max(R_est, R_est_1)]))
        )
    else:
        popt_p, _ = scipy.optimize.curve_fit(
        lambda t, tau_D: get_voltage(t, tau_D, R_est), t_pulse_data_charge, voltage_data, p0=[tau_D_est], bounds=(0, np.inf)
        )
        popt_p = [popt_p[0], R_est]

    print(f"Initial tau_Dp: {tau_D_est}, fitted tau_Dp: {popt_p[0]}")
    print(f"Initial Rp: {R_est}, fitted Rp: {popt_p[1]}")

    t_est_charge.append(popt_p[0])
    Rp_est_charge.append(popt_p[1])

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
fig1, ax1 = plt.subplots(5, 4, sharex=True, figsize=(3, 2))
ax1 = ax1.flatten()
for i in range(20):
    t_pulse_data_discharge = np.linspace(1, tpulse_discharge[i], 1000)

    voltage_sim = get_sim_voltage(t_pulse_data_discharge, t_est[i], Rp_est[i], final_stochiometry[i], 
                                  initial_stochiometry[i], average_current[i], tpulse_discharge[i])
    
    ocv_sim = get_sim_ocv(t_pulse_data_discharge, t_est[i], Rp_est[i], final_stochiometry[i], 
                                  initial_stochiometry[i], average_current[i], tpulse_discharge[i])
    pulse = discharge_pulses[i]
    before_pulse = before_discharge_pulses[i]
    voltage_diff = voltage_sim - ocv_sim

    true_v_interp = scipy.interpolate.interp1d(
        pulse["processed_time_s"], pulse["voltage"]
        )

    true_voltage_diff = true_v_interp(t_pulse_data_discharge) - ocv_sim
    ax[i].plot(pulse["processed_time_s"], pulse["voltage"], label="Data")
    ax[i].plot(before_pulse["processed_time_s"], before_pulse["voltage"], color="k")
    ax[i].plot(t_pulse_data_discharge, voltage_sim, linestyle="--", label="Fit")
    ax[i].plot(t_pulse_data_discharge, ocv_sim, linestyle=":", label="OCV")
    
    ax1[i].plot(t_pulse_data_discharge, voltage_diff, linestyle=":", label="Voltage Difference")
    ax1[i].plot(t_pulse_data_discharge, true_voltage_diff, linestyle=":", label="True Voltage Difference")
    # ax1[i].set_ylabel("Potential [V]")

    # ax[i].legend()
    # ax[i].set_xlabel("Time [s]")

    ax1[i].set_xlim([-30, max(t_pulse_data_discharge)])

    ax1[i].grid(True)

fig_charge, ax_charge = plt.subplots(5, 4, sharex=True, figsize=(3, 2))
ax_charge = ax_charge.flatten()
fig1_charge, ax1_charge = plt.subplots(5, 4, sharex=True, figsize=(3, 2))
ax1_charge = ax1_charge.flatten()
for i in range(20):
    t_pulse_data_charge = np.linspace(1, tpulse_charge[i], 1000)

    voltage_sim = get_sim_voltage(t_pulse_data_charge, t_est_charge[i], Rp_est_charge[i], final_stochiometry_charge[i], 
                                  initial_stochiometry_charge[i], average_current_charge[i], tpulse_charge[i])
    
    ocv_sim = get_sim_ocv(t_pulse_data_charge, t_est_charge[i], Rp_est_charge[i], final_stochiometry_charge[i], 
                                  initial_stochiometry_charge[i], average_current_charge[i], tpulse_charge[i])
    pulse_charge = charge_pulses[i]
    before_pulse_charge = before_charge_pulses[i]
    voltage_diff = voltage_sim - ocv_sim

    true_v_interp = scipy.interpolate.interp1d(
        pulse_charge["processed_time_s"], pulse_charge["voltage"], kind="linear", fill_value="extrapolate"
        )

    true_voltage_diff = true_v_interp(t_pulse_data_charge) - ocv_sim
    ax_charge[i].plot(pulse_charge["processed_time_s"], pulse_charge["voltage"], label="Data")
    ax_charge[i].plot(before_pulse_charge["processed_time_s"], before_pulse_charge["voltage"], color="k")
    ax_charge[i].plot(t_pulse_data_charge, voltage_sim, linestyle="--", label="Fit")
    ax_charge[i].plot(t_pulse_data_charge, ocv_sim, linestyle=":", label="OCV")
    
    ax1_charge[i].plot(t_pulse_data_charge, voltage_diff, linestyle=":", label="Voltage Difference")
    ax1_charge[i].plot(t_pulse_data_charge, true_voltage_diff, linestyle=":", label="True Voltage Difference")
    # ax1[i].set_ylabel("Potential [V]")

    # ax[i].legend()
    # ax[i].set_xlabel("Time [s]")

    ax1_charge[i].set_xlim([-30, max(t_pulse_data_charge)])

    ax1_charge[i].grid(True)

##############################################################################################################
# # Plotting tau vs initial stoichiometry
# fig_tau_vs_stoich, ax_tau_vs_stoich = plt.subplots()
# ax_tau_vs_stoich.scatter(initial_stochiometry, t_est, label='Data Points')
# ax_tau_vs_stoich.plot(initial_stochiometry, t_est, '-', color='red', label='Connecting Line')
# ax_tau_vs_stoich.set_xlabel('Initial Stoichiometry')
# ax_tau_vs_stoich.set_ylabel('Tau')
# ax_tau_vs_stoich.legend()


# Calculate 1/tau values
particle_radius = 6e-6
inv_tau_values = 1 / np.array(t_est) 


# # # Plotting 1/tau vs initial stoichiometry
# # fig_inv_tau_vs_stoich, ax_inv_tau_vs_stoich = plt.subplots()
# # ax_inv_tau_vs_stoich.scatter(initial_stochiometry, inv_tau_values, label='Data Points')
# # ax_inv_tau_vs_stoich.plot(initial_stochiometry, inv_tau_values, '-', color='green', label='Connecting Line')
# # ax_inv_tau_vs_stoich.set_xlabel('Initial Stoichiometry')
# # ax_inv_tau_vs_stoich.set_ylabel('1/Tau')
# # ax_inv_tau_vs_stoich.legend()

# Calculate average stoichiometry
average_stoichiometry = (np.array(initial_stochiometry) + np.array(final_stochiometry)) / 2

# # # Plotting 1/tau vs average stoichiometry
# # fig_inv_tau_vs_avg_stoich, ax_inv_tau_vs_avg_stoich = plt.subplots()
# # ax_inv_tau_vs_avg_stoich.scatter(average_stoichiometry, inv_tau_values, label='Data Points')
# # ax_inv_tau_vs_avg_stoich.plot(average_stoichiometry, inv_tau_values, '-', color='green', label='Connecting Line')
# # ax_inv_tau_vs_avg_stoich.set_xlabel('Average Stoichiometry')
# # ax_inv_tau_vs_avg_stoich.set_ylabel('1/Tau')
# # ax_inv_tau_vs_avg_stoich.legend()

# Plotting 1/tau vs both initial and average stoichiometry
fig_inv_tau_vs_stoich, ax_inv_tau_vs_stoich = plt.subplots()
ax_inv_tau_vs_stoich.scatter(initial_stochiometry, inv_tau_values, label='Initial Stoichiometry')
ax_inv_tau_vs_stoich.plot(initial_stochiometry, inv_tau_values, '-', color='green', label='Connecting Line')
ax_inv_tau_vs_stoich.scatter(average_stoichiometry, inv_tau_values, label='Average Stoichiometry')
ax_inv_tau_vs_stoich.plot(average_stoichiometry, inv_tau_values, '-', color='red', label='Connecting Line')
ax_inv_tau_vs_stoich.set_xlabel('Stoichiometry')
ax_inv_tau_vs_stoich.set_ylabel('1/Tau')
ax_inv_tau_vs_stoich.legend()

plt.show()

print("Hi")




