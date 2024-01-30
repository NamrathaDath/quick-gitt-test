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

# Lists to store diffusion coefficients
diffusion_coefficients = []

def process_pulse_data(step_numbers, df, pulse_df):
    """
    Process pulse data and calculate various parameters.

    Args:
        step_numbers (list): List of step numbers.
        df (pandas.DataFrame): DataFrame containing data for each step.
        pulse_df (pandas.DataFrame): DataFrame containing pulse data.

    Returns:
        tuple: A tuple containing various calculated parameters including voltage values, time values,
               pulse duration, initial and final current, before pulse data, pulse data, initial and final stochiometry,
               voltage drop and rise, average current, and internal resistance.
    """
    voltage1, voltage2, voltage3, voltage4, voltage_last_point = [], [], [], [], []
    time1, time2, time3, time4 = [], [], [], []
    tpulse = []
    initial_current, final_current = [], []
    before_pulses, pulses = [], []

    for i in step_numbers:
        Before_pulse = df[df["step_number"] == i - 1]
        pulse = pulse_df[pulse_df["step_number"] == i]
        After_pulse = df[df["step_number"] == i + 1]

        voltage1.append(Before_pulse["voltage"].iloc[-1])
        voltage2.append(pulse["voltage"].iloc[0])
        voltage3.append(pulse["voltage"].iloc[-1])
        voltage_last_point.append(After_pulse["voltage"].iloc[-1])
        voltage4.append(After_pulse["voltage"].iloc[0])

        time1.append(Before_pulse["total_time_millis"].iloc[-1])
        time2.append(pulse["total_time_millis"].iloc[0])
        time3.append(pulse["total_time_millis"].iloc[-1])
        time4.append(After_pulse["total_time_millis"].iloc[0])

        pulse_start_time = pulse["total_time_millis"].iloc[0]

        pulse["processed_time_s"] = (pulse["total_time_millis"] - pulse_start_time) * 1e-3
        tpulse.append(pulse["processed_time_s"].max())

        initial_current.append(pulse["current"].iloc[1])
        final_current.append(pulse["current"].iloc[-1])

        Before_pulse["processed_time_s"] = (Before_pulse["total_time_millis"] - pulse_start_time) * 1e-3
        before_pulses.append(Before_pulse)
        pulses.append(pulse)
    
    # Calculate initial and final stochiometry
    initial_stochiometry = U_inv(voltage1)
    final_stochiometry = U_inv(voltage_last_point)

    # Calculate voltage drop and rise
    initial_delta_v = np.array(voltage2) - np.array(voltage1)
    final_delta_v = np.array(voltage3) - np.array(voltage4)

    # Calculate average current 
    average_current = (np.array(initial_current) + np.array(final_current)) / 2

    # Calculate internal resistance
    internal_resistance_1 = np.array(initial_delta_v)/np.array(average_current)
    internal_resistance_2 = np.array(final_delta_v)/np.array(average_current)

    # Print or use the calculated values
    for i, step_number in enumerate(step_numbers):
        print(f"Step number {step_number}: Initial Delta V = {initial_delta_v[i]}, Final Delta V = {final_delta_v[i]}, "
              f"Average Current = {average_current[i]}, Initial Resistance change = {internal_resistance_1[i]}, "
              f"Final resistance change = {internal_resistance_2[i]}, Pulse Time = {tpulse[i]}")
        
    return voltage1, voltage2, voltage3, voltage4, voltage_last_point, time1, time2, time3, time4, tpulse, initial_current, final_current, before_pulses, pulses, initial_stochiometry, final_stochiometry, initial_delta_v, final_delta_v, average_current, internal_resistance_1, internal_resistance_2


# Process discharge pulses
(v1_discharge, v2_discharge, v3_discharge, v4_discharge, v4_discharge_last_point, t1_discharge, t2_discharge,
 t3_discharge, t4_discharge, tpulse_discharge, initial_current, final_current, before_discharge_pulses, discharge_pulses, initial_stochiometry_discharge, final_stochiometry_discharge, initial_delta_v_discharge, final_delta_v_discharge, average_current_discharge, internal_resistance_drop_discharge, internal_resistance_rise_discharge) = process_pulse_data(
    discharge_step_numbers, df, discharge_df
)

# Process charge pulses
(v1_charge, v2_charge, v3_charge, v4_charge, v4_charge_last_point, t1_charge, t2_charge, t3_charge, t4_charge,
 tpulse_charge, initial_current_charge, final_current_charge, before_charge_pulses, charge_pulses, initial_stochiometry_charge, final_stochiometry_charge, initial_delta_v_charge, final_delta_v_charge, average_current_charge, internal_resistance_rise_charge, internal_resistance_drop_charge) = process_pulse_data(
    charge_step_numbers, df, charge_df
)

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(df["total_time_millis"], df["current"])
ax[0].scatter(t2_discharge, initial_current, color='blue', marker='o')
ax[0].scatter(t2_charge, initial_current_charge, color='blue', marker='o')
ax[0].scatter(t3_discharge, final_current, color='orange', marker='o')
ax[0].scatter(t3_charge, final_current_charge, color='orange', marker='o')
ax[0].set_ylabel('Current (A)')

ax[1].plot(df["total_time_millis"], df["voltage"])
ax[1].scatter(t1_discharge, v1_discharge, color='red', marker='o')
ax[1].scatter(t1_charge, v1_charge, color='red', marker='o')
ax[1].scatter(t2_discharge, v2_discharge, color='blue', marker='o')
ax[1].scatter(t2_charge, v2_charge, color='blue', marker='o')
ax[1].scatter(t3_discharge, v3_discharge, color='orange', marker='o')
ax[1].scatter(t3_charge, v3_charge, color='orange', marker='o')
ax[1].scatter(t4_discharge, v4_discharge, color='black', marker='o')
ax[1].scatter(t4_charge, v4_charge, color='black', marker='o')
ax[1].set_ylabel('Voltage (V)')



t_est, t_est_charge = [], []
Rp_est, Rp_est_charge = [], []


# for i in range(len(tpulse_discharge)):

#     # Calculate tau and Rp
#     # def get_voltage_sand(t, tau_D, R):
#     #     sto_surf = (2 / (3 * np.sqrt(np.pi))) * (final_stochiometry_discharge[i] - initial_stochiometry_discharge[i]) / tpulse_discharge[i] * np.sqrt(
#     #         t * tau_D
#     #     ) + initial_stochiometry_discharge[i]
#     #     V = U(sto_surf) + R * average_current_discharge[i]
#     #     return V
    
#     def get_voltage(t, tau_D, R):
#         t_tilde = t / tau_D
#         sto_surf = 1/3 * (final_stochiometry_discharge[i] - initial_stochiometry_discharge[i]) * tau_D / tpulse_discharge[i] * c_tilde(t_tilde) + initial_stochiometry_discharge[i]
#         V = U(sto_surf) + R * average_current_discharge[i]
#         return V

#     t_pulse_data_discharge = np.linspace(1, tpulse_discharge[i], 1000)

#     pulse = discharge_pulses[i]
#     v_pulse = scipy.interpolate.interp1d(
#         pulse["processed_time_s"], pulse["voltage"]
#         )

#     voltage_data = v_pulse(t_pulse_data_discharge)

#     # curve fit the vp data
#     R_est = initial_delta_v_discharge[i] / average_current_discharge[i]
#     R_est_1 = final_delta_v_discharge[i] / average_current_discharge[i]
#     tau_D_est = 3600


#     if fit_Rp:
#         popt_p, _ = scipy.optimize.curve_fit(
#         get_voltage, t_pulse_data_discharge, voltage_data, p0=[tau_D_est, R_est], bounds=(np.array([0,0]), np.array([np.inf, np.inf]))
#         )
#     else:
#         popt_p, _ = scipy.optimize.curve_fit(
#         lambda t, tau_D: get_voltage(t, tau_D, R_est), t_pulse_data_discharge, voltage_data, p0=[tau_D_est], bounds=(0, np.inf)
#         )
#         popt_p = [popt_p[0], R_est]

#     print(f"Initial tau_Dp: {tau_D_est}, fitted tau_Dp: {popt_p[0]}")
#     print(f"Initial Rp: {R_est}, fitted Rp: {popt_p[1]}")

#     t_est.append(popt_p[0])
#     Rp_est.append(popt_p[1])



for i in range(len(tpulse_charge)):

    # Calculate tau and Rp
    # def get_voltage_sand(t, tau_D, R):
    #     sto_surf = (2 / (3 * np.sqrt(np.pi))) * (final_stochiometry_charge[i] - initial_stochiometry_charge[i]) / tpulse_charge[i] * np.sqrt(
    #         t * tau_D
    #     ) + initial_stochiometry_charge[i]
    #     V = U(sto_surf) + R * average_current_charge[i]
    #     return V

    

    def get_voltage(t, tau_D, R):
        t_tilde = t / tau_D
        sto_surf = 1/3 * (final_sto - initial_sto) * tau_D / tpulse * c_tilde(t_tilde) + initial_sto
        V = U(sto_surf) + R * average_current
        return V
    

    t_pulse_data_discharge = np.linspace(1, tpulse_discharge[i], 1000)
    t_pulse_data_charge = np.linspace(1, tpulse_charge[i], 1000)

    pulse_discharge = discharge_pulses[i]
    v_pulse_discharge = scipy.interpolate.interp1d(
        pulse_discharge["processed_time_s"], pulse_discharge["voltage"], kind="linear", fill_value="extrapolate"
        )
    pulse_charge = charge_pulses[i]
    v_pulse_charge = scipy.interpolate.interp1d(
        pulse_charge["processed_time_s"], pulse_charge["voltage"], kind="linear", fill_value="extrapolate"
        )

    voltage_data_discharge = v_pulse_discharge(t_pulse_data_discharge)
    voltage_data_charge = v_pulse_charge(t_pulse_data_charge)

    # curve fit the vp data
    R_est_discharge = final_delta_v_discharge[i] / average_current_discharge[i]
    R_est_1_discharge = initial_delta_v_discharge[i] / average_current_discharge[i]
    R_est_charge = final_delta_v_charge[i] / average_current_charge[i]
    R_est_1_charge = initial_delta_v_charge[i] / average_current_charge[i]
    tau_D_est = 3600


    if fit_Rp:
        final_sto = final_stochiometry_discharge[i]
        initial_sto = initial_stochiometry_discharge[i]
        average_current = average_current_discharge[i]
        tpulse = tpulse_discharge[i]

        popt_p, _ = scipy.optimize.curve_fit(
        get_voltage, t_pulse_data_discharge, voltage_data_discharge, p0=[tau_D_est, R_est_discharge], bounds=(np.array([0,0]), np.array([np.inf, np.inf]))
        )

        final_sto = final_stochiometry_charge[i]
        initial_sto = initial_stochiometry_charge[i]
        average_current = average_current_charge[i]
        tpulse = tpulse_charge[i]

        popt_p_charge, _ = scipy.optimize.curve_fit(
        get_voltage, t_pulse_data_charge,voltage_data_charge, p0=[tau_D_est, R_est_charge], bounds=(np.array([0,0]), np.array([np.inf,np.inf]))
        )
    else:
        popt_p, _ = scipy.optimize.curve_fit(
        lambda t, tau_D: get_voltage(t, tau_D, R_est_discharge), t_pulse_data_discharge, voltage_data_discharge, p0=[tau_D_est], bounds=(0, np.inf)
        )
        popt_p = [popt_p[0], R_est_discharge]
        popt_p_charge, _ = scipy.optimize.curve_fit(
        lambda t, tau_D: get_voltage(t, tau_D, R_est_charge), t_pulse_data_charge, voltage_data_charge, p0=[tau_D_est], bounds=(0, np.inf)
        )
        popt_p_charge = [popt_p_charge[0], R_est_charge]

    print(f"Initial tau_Dp: {tau_D_est}, fitted tau_Dp: {popt_p[0]}")
    print(f"Initial Rp: {R_est_discharge}, fitted Rp: {popt_p[1]}")
    print(f"Initial tau_Dp: {tau_D_est}, fitted tau_Dp: {popt_p_charge[0]}")
    print(f"Initial Rp: {R_est_charge}, fitted Rp: {popt_p_charge[1]}")

    t_est_charge.append(popt_p_charge[0])
    Rp_est_charge.append(popt_p_charge[1])

    t_est.append(popt_p[0])
    Rp_est.append(popt_p[1])
    ##############################################################################################################
    
    # final_sto = final_stochiometry_charge[i]
    # initial_sto = initial_stochiometry_charge[i]
    # average_current = average_current_charge[i]
    # tpulse = tpulse_charge[i]

    # if fit_Rp:
    #     popt_p_charge, _ = scipy.optimize.curve_fit(
    #     get_voltage, t_pulse_data_charge,voltage_data, p0=[tau_D_est, R_est], bounds=(np.array([0,0]), np.array([np.inf,np.inf]))
    #     )
    # else:
    #     popt_p_charge, _ = scipy.optimize.curve_fit(
    #     lambda t, tau_D: get_voltage(t, tau_D, R_est_charge), t_pulse_data_charge, voltage_data, p0=[tau_D_est], bounds=(0, np.inf)
    #     )
    #     popt_p_charge = [popt_p_charge[0], R_est]

    # print(f"Initial tau_Dp: {tau_D_est}, fitted tau_Dp: {popt_p_charge[0]}")
    # print(f"Initial Rp: {R_est_charge}, fitted Rp: {popt_p_charge[1]}")

    # t_est_charge.append(popt_p_charge[0])
    # Rp_est_charge.append(popt_p_charge[1])

# def get_sim_voltage_sands(t, tau_D, R, final_stochiometry, initial_stochiometry, average_current, tpulse):
#     sto_surf = (2 / (3 * np.sqrt(np.pi))) * (final_stochiometry - initial_stochiometry) / tpulse * np.sqrt(
#             t * tau_D
#         ) + initial_stochiometry
#     V = U(sto_surf) + R * average_current
#     return V

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

    voltage_sim = get_sim_voltage(t_pulse_data_discharge, t_est[i], Rp_est[i], final_stochiometry_discharge[i], 
                                  initial_stochiometry_discharge[i], average_current_discharge[i], tpulse_discharge[i])
    
    ocv_sim = get_sim_ocv(t_pulse_data_discharge, t_est[i], Rp_est[i], final_stochiometry_discharge[i], 
                                  initial_stochiometry_discharge[i], average_current_discharge[i], tpulse_discharge[i])
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
inv_tau_values = 1 / np.array(t_est) * particle_radius ** 2


# # # Plotting 1/tau vs initial stoichiometry
# # fig_inv_tau_vs_stoich, ax_inv_tau_vs_stoich = plt.subplots()
# # ax_inv_tau_vs_stoich.scatter(initial_stochiometry, inv_tau_values, label='Data Points')
# # ax_inv_tau_vs_stoich.plot(initial_stochiometry, inv_tau_values, '-', color='green', label='Connecting Line')
# # ax_inv_tau_vs_stoich.set_xlabel('Initial Stoichiometry')
# # ax_inv_tau_vs_stoich.set_ylabel('1/Tau')
# # ax_inv_tau_vs_stoich.legend()

def calculate_average_stoichiometry(initial_stochiometry, final_stochiometry):
    return (np.array(initial_stochiometry) + np.array(final_stochiometry)) / 2

# Calculate average stoichiometry
average_stoichiometry = calculate_average_stoichiometry(initial_stochiometry_discharge, final_stochiometry_discharge)
average_stoichiometry_charge = calculate_average_stoichiometry(initial_stochiometry_charge, final_stochiometry_charge)



# # # Plotting 1/tau vs average stoichiometry
# # fig_inv_tau_vs_avg_stoich, ax_inv_tau_vs_avg_stoich = plt.subplots()
# # ax_inv_tau_vs_avg_stoich.scatter(average_stoichiometry, inv_tau_values, label='Data Points')
# # ax_inv_tau_vs_avg_stoich.plot(average_stoichiometry, inv_tau_values, '-', color='green', label='Connecting Line')
# # ax_inv_tau_vs_avg_stoich.set_xlabel('Average Stoichiometry')
# # ax_inv_tau_vs_avg_stoich.set_ylabel('1/Tau')
# # ax_inv_tau_vs_avg_stoich.legend()

# Plotting 1/tau vs both initial and average stoichiometry
fig_inv_tau_vs_stoich, ax_inv_tau_vs_stoich = plt.subplots()
ax_inv_tau_vs_stoich.scatter(initial_stochiometry_discharge, inv_tau_values, label='Initial Stoichiometry')
ax_inv_tau_vs_stoich.plot(initial_stochiometry_discharge, inv_tau_values, '-', color='green', label='Connecting Line')
ax_inv_tau_vs_stoich.scatter(average_stoichiometry, inv_tau_values, label='Average Stoichiometry')
ax_inv_tau_vs_stoich.plot(average_stoichiometry, inv_tau_values, '-', color='red', label='Connecting Line')
ax_inv_tau_vs_stoich.set_xlabel('Stoichiometry')
ax_inv_tau_vs_stoich.set_ylabel('1/Tau (Discharge)')
#ax_inv_tau_vs_stoich.set_ylim([0, 0.0004])
ax_inv_tau_vs_stoich.legend()

inv_tau_values_charge = 1 / np.array(t_est_charge) * particle_radius ** 2

# Plotting 1/tau vs both initial and average stoichiometry
fig_inv_tau_vs_stoich_charge, ax_inv_tau_vs_stoich_charge = plt.subplots()
ax_inv_tau_vs_stoich_charge.scatter(initial_stochiometry_charge, inv_tau_values_charge, label='Initial Stoichiometry')
ax_inv_tau_vs_stoich_charge.plot(initial_stochiometry_charge, inv_tau_values_charge, '-', color='green', label='Connecting Line')
ax_inv_tau_vs_stoich_charge.scatter(average_stoichiometry_charge, inv_tau_values_charge, label='Average Stoichiometry')
ax_inv_tau_vs_stoich_charge.plot(average_stoichiometry_charge, inv_tau_values_charge, '-', color='red', label='Connecting Line')
ax_inv_tau_vs_stoich_charge.set_xlabel('Stoichiometry')
ax_inv_tau_vs_stoich_charge.set_ylabel('1/Tau (Charge)')
#ax_inv_tau_vs_stoich_charge.set_ylim([0, 0.0004])
ax_inv_tau_vs_stoich_charge.legend()



plt.show()

print("Hi")




