import valibrary as vl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate

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
instantaneous_rise_voltage = []

initial_time = []
instantaneous_drop_time = []
final_time = []
instantaneous_rise_time = []

for i in discharge_step_numbers:
    Before_pulse = df[df["step_number"] == i-1] #dataframe to get the before pulse
    pulse = discharge_df[discharge_df["step_number"] == i]
    After_pulse = df[df["step_number"] == i+1]
    
    initial_voltage.append(Before_pulse["voltage"].iloc[-1])
    instantaneous_drop_voltage.append(pulse["voltage"].iloc[1])
    final_voltage.append(pulse["voltage"].iloc[-1])
    instantaneous_rise_voltage.append(After_pulse["voltage"].iloc[0])

    # initial_time.append(pulse["total_time_millis"].iloc[0])
    initial_time.append(Before_pulse["total_time_millis"].iloc[-1])
    instantaneous_drop_time.append(pulse["total_time_millis"].iloc[1])
    final_time.append(pulse["total_time_millis"].iloc[-1])
    instantaneous_rise_time.append(After_pulse["total_time_millis"].iloc[0])



# step_df = df.groupby("step_number").max() 
# step_numbers = step_df.index.to_numpy()

# pulses = []

# for i, sn in enumerate(step_numbers): 

#     if sn in discharge_step_numbers: 
#         prev_sn = step_numbers[i-1]
#         next_sn = step_numbers[i+1]

#         idx = df["step_number"].isin([prev_sn, sn, next_sn])

#         df_prev = df[df["step_number"]==prev_sn]
#         df_pulse = df[df["step_number"]==sn]
#         df_next = df[df["step_number"]==next_sn]

#         t_pulse_start = df_pulse["total_time_millis"].iloc[0]
#         df_prev["pulse_time_s"] = (df_prev["total_time_millis"] - t_pulse_start) * 1e-3
#         df_pulse["pulse_time_s"] = (df_pulse["total_time_millis"] - t_pulse_start) * 1e-3
#         df_next["pulse_time_s"] = (df_next["total_time_millis"] - t_pulse_start) * 1e-3

#         t_pulse_end = df_next["total_time_millis"].iloc[0]
#         df_prev["time_to_pulse_end_s"] = (df_prev["total_time_millis"] - t_pulse_end) * 1e-3
#         df_pulse["time_to_pulse_end_s"] = (df_pulse["total_time_millis"] - t_pulse_end) * 1e-3
#         df_next["time_to_pulse_end_s"] = (df_next["total_time_millis"] - t_pulse_end) * 1e-3

#         df_joined = pd.concat([df_prev, df_pulse, df_next])     #df_joined = pd.concat([df_prev, df_pulse, df_next])

#         pulses.append(
#             {
#                 "prev": df_prev, 
#                 "pulse": df_pulse,
#                 "next": df_next, 
#                 "joined": df_joined
#             }
#         )

# pulses = pulses[:2]

# pulse = pulses[0]

# initial_voltage = pulse["prev"]["voltage"].iloc[-1]

# voltage_interp = scipy.interpolate.interp1d(pulse["pulse"]["pulse_time_s"], pulse["pulse"]["voltage"])
# voltage_1s = voltage_interp(1)

# initial_voltage_drop = voltage_1s - initial_voltage

# voltage_interp = scipy.interpolate.interp1d(pulse["pulse"]["time_to_pulse_end_s"], pulse["pulse"]["voltage"])
# final_pulse_voltage = voltage_interp(-1)

# voltage_interp = scipy.interpolate.interp1d(pulse["next"]["time_to_pulse_end_s"], pulse["next"]["voltage"])
# first_rest_voltage = voltage_interp(0)

# final_voltage_drop = first_rest_voltage - final_pulse_voltage

# print("Inital drop")
# print(initial_voltage_drop)
# print("Final drop")
# print(final_voltage_drop)


# instantaneous_drop_indices = []
# instantaneous_rise_indices = []
      
# for pulse in pulses: 

#     fig, ax = plt.subplots(2, 1)

#     df = pulse["joined"]
#     current_diff = np.diff(df["current"])
#     drop_indices = np.where(current_diff < 0.0)[0]
#     rise_indices = np.where(current_diff >= 0.0001)[0]
#     '''next_indices = drop_indices + 1
#     indices_with_next = np.concatenate([drop_indices, next_indices])
#     sorted_indices = np.sort(indices_with_next)'''
#     instantaneous_drop_indices.extend(drop_indices)
#     instantaneous_rise_indices.extend(rise_indices)
    
    
#     ax[0].plot(df["total_time_millis"], df["current"])
#     ax[0].set_ylabel('Current (A)')


#     ax[1].plot(df["total_time_millis"], df["voltage"])
#     ax[1].set_ylabel('Voltage (V)')
    
    
    
#     '''ax[2].plot(df["total_time_millis"], np.gradient(df["voltage"], df["total_time_millis"]))
#     ax[2].set_ylabel('Voltage Derivative')'''

# Calculate voltage drop and rise
voltage_drop = np.array(instantaneous_drop_voltage) - np.array(initial_voltage)
voltage_rise = np.array(final_voltage) - np.array(instantaneous_rise_voltage)

# Print or use the calculated values as needed
for i, step_number in enumerate(discharge_step_numbers):
    print(f"Pulse {step_number}: Voltage Drop = {voltage_drop[i]}, Voltage Rise = {voltage_rise[i]}")


# plt.figure(figsize=(10, 6))
# plt.plot(df["total_time_millis"], df["voltage"], label="Voltage")
# plt.scatter(df.iloc[instantaneous_drop_indices]["total_time_millis"], df.iloc[instantaneous_drop_indices]["voltage"], color='red', label="Instantaneous Drop")
# plt.scatter(df.iloc[instantaneous_rise_indices]["total_time_millis"], df.iloc[instantaneous_rise_indices]["voltage"], color='blue', label="Instantaneous Rise")
# plt.xlabel('Time (ms)')
# plt.ylabel('Voltage (V)')
# plt.legend()
# plt.title('Instantaneous Voltage Drop and Rise')

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(actual_df["total_time_millis"], actual_df["current"])
ax[0].set_ylabel('Current (A)')


ax[1].plot(actual_df["total_time_millis"], actual_df["voltage"])
ax[1].scatter(initial_time, initial_voltage, color='red', marker='o')
ax[1].scatter(instantaneous_drop_time, instantaneous_drop_voltage, color='blue', marker='o')
ax[1].scatter(final_time, final_voltage, color='yellow', marker='o')
ax[1].scatter(instantaneous_rise_time, instantaneous_rise_voltage, color='black', marker='o')

ax[1].set_ylabel('Voltage (V)')




plt.show()
print("Hi")




