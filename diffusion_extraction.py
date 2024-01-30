import valibrary as vl
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

fit_Rp = False

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

# Input the GITT data
filepath = "231030_GPA-A-B1_GITT_(T25_NV008)_A3122_C12.txt"
converter = vl.BiologicConverter(filepath)

raw_df = converter.to_cloud()

#Plot raw data
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 6))

time_s = raw_df["total_time_millis"]

ax[0].plot(time_s, raw_df["current"])
ax[1].plot(time_s, raw_df["voltage_anode"])
ax[2].plot(time_s, raw_df["voltage_cathode"])

ax[3].plot(time_s, raw_df["step_number"])

ax[0].set_ylabel("Current [A]")
ax[1].set_ylabel("Anode potential [V]")
ax[2].set_ylabel("Cathode potential [V]")
ax[3].set_ylabel("Step number")

ax[-1].set_xlabel("Time [h]")

for a in ax:
    a.grid(True)

plt.tight_layout()
# plt.show()

#extract pulse data
pulse_steps = [5, 6, 7]
# pulse_steps = [21, 22, 23]
# pulse_steps = [65, 66, 67]
# pulse_steps = [41, 42, 43]
# pulse_steps = [46, 47, 48]

df = raw_df[raw_df["step_number"].isin(pulse_steps)]
before_df = raw_df[raw_df["step_number"] == pulse_steps[0]]
pulse_df = raw_df[raw_df["step_number"] == pulse_steps[1]]
after_df = raw_df[raw_df["step_number"] == pulse_steps[2]]

U_init = before_df["voltage_cathode"].iloc[-1]
U_final = after_df["voltage_cathode"].iloc[-1]

# get start time of the pulse
t0 = pulse_df["total_time_millis"].iloc[0]
pulse_df["pulse_time_s"] = (pulse_df["total_time_millis"] - t0) * 1e-3
df["pulse_time_s"] = (df["total_time_millis"] - t0) * 1e-3

df = df["pulse_time_s"] > -30

# interpolate the pulse_voltage
v_pulse = scipy.interpolate.interp1d(
    pulse_df["pulse_time_s"], pulse_df["voltage_cathode"]
)

sto_init = U_inv(U_init)
sto_final = U_inv(U_final)

# find the initial voltage drop
v_drop = v_pulse(1) - U_init

# get the current from the pulse
current = pulse_df["current"].iloc[3:-3].mean()
t_pulse = pulse_df["pulse_time_s"].max()


def get_voltage(t, tau_D, R):
    sto_surf = (2 / (3 * np.sqrt(np.pi))) * (sto_final - sto_init) / t_pulse * np.sqrt(
        t * tau_D
    ) + sto_init
    V = U(sto_surf) + R * current
    return V


t_pulse_data = np.linspace(1, t_pulse, 1000)
voltage_data = v_pulse(t_pulse_data)

# curve fit the vp data
R_est = v_drop / current
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

voltage_sim = get_voltage(t_pulse_data, popt_p[0], popt_p[1])


fig, ax = plt.subplots(1, 1, sharex=True, figsize=(9, 6))

ax.plot(pulse_df["pulse_time_s"], pulse_df["voltage_cathode"], label="Data")
ax.plot(t_pulse_data, voltage_sim, linestyle="--", label="Fit")

ax.set_ylabel("Cathode potential [V]")

ax.legend()
ax.set_xlabel("Time [s]")

ax.grid(True)

plt.show()
