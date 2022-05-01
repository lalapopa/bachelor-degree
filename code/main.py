# TODO:
# calculate climb
import os
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere
from matplotlib import rc
from datetime import datetime

from lerp import linear1d
from DataHandler import DataHandler as dh
from plane_data import PlaneData
from aerodynamics_data import AerodynamicsData
from formulas import Formulas as eq


class Calculation:
    def __init__(self, mass, plane_char, H):
        self.ad = AerodynamicsData()
        self.plane_char = plane_char
        self.MACH_calc = np.arange(0.3, 0.95, 0.001)
        self.g = 9.81

        self.mass = mass
        self.H = float(H)  # ambiance package can't take np.int16 type
        air_H = Atmosphere(self.H)
        air_11 = Atmosphere(11000)
        Ro = air_H.density[0]
        self.a_sos = air_H.speed_of_sound[0]

        self.V_calc = eq.V_speed(self.MACH_calc, self.a_sos)
        q = eq.q_dynamic_pressure(self.V_calc, Ro)

        Cy = eq.my_C_y_n(
            self.mass, self.g, self.plane_char.WING_AREA, Ro, self.MACH_calc, self.a_sos
        )
        A = self.ad.A_value(self.MACH_calc)
        Cxm = self.ad.Cxm_value(self.MACH_calc)
        Cym = self.ad.Cym_value(self.MACH_calc)
        Cx = eq.C_x_n_drag_coefficient(Cxm, A, Cy, Cym)

        K = eq.K_n_lift_to_drag_ratio(Cy, Cx)

        self.P_potr = eq.P_potr_equation(self.mass, self.g, K)
        tilda_P = np.array([self.ad.PTilda(M, self.H) for M in self.MACH_calc])
        self.P_rasp = eq.P_rasp_equation(
            self.plane_char.OTN_P_0,
            self.mass,
            self.g,
            tilda_P,
            self.H,
            air_11.pressure[0],
            air_H.pressure[0],
        )

    def find_min_fuel_consumption(self):
        P_diff = self.P_rasp - self.P_potr
        if self._can_fly(P_diff):
            tilda_R = eq.tilda_R_equation(self.P_potr, self.P_rasp)
            Ce_tilda = [self.ad.CeTilda(M, self.H) for M in self.MACH_calc]
            Ce_dr = self.ad.Ce_dr_value(tilda_R)
            Ce = eq.Ce_equation(self.plane_char.CE_0, Ce_tilda, Ce_dr)
            q_chas = eq.q_ch_hour_consumption(Ce, self.P_potr)

            q_km = eq.q_km_range_consumption(q_chas, self.V_calc)

            min_km_fuel_index = dh.get_min_or_max(q_km)
            mach_min_fuel = self.V_calc[min_km_fuel_index] / self.a_sos

            return (
                q_km[min_km_fuel_index],
                mach_min_fuel,
                self.V_calc[min_km_fuel_index],
            )
        #        return q_chas[min_km_fuel_index], mach_min_fuel, V_calc[min_km_fuel_index]
        else:
            return False

    def _can_fly(self, thrust_diff):
        for i in thrust_diff:
            if np.sign(i) > 0:
                return True
        return False

    def find_Vy_speeds(self):
        return self.V_calc*(self.P_rasp - self.P_potr)/(self.mass*self.g)


def L_range(mass):
    H_array = np.array([7500])
    q_km_min = []
    for H in H_array:
        calc = Calculation(mass, il_76, H)
        result = calc.find_min_fuel_consumption()
        if result:
            q_km, _, _ = result
            q_km_min.append(q_km)
    min_index = dh.get_min_or_max(q_km_min)
    result = 1 / q_km_min[min_index]
    return result


def to_height_mach_table(H, M, q_km, V, index):
    table = []
    for i, val in enumerate(H):
        if index == i:
            add_text = "\\cellcolor{green}"
        else:
            add_text = ""
        only_H = {}
        only_H["$M$"] = str(f"{M[i]:.3f} {add_text}")
        only_H["$q_{km}$"] = str(f"{q_km[i]:.2f} {add_text}")
        only_H["$V$"] = str(f"{V[i]:.0f} {add_text}")
        only_H["$H$"] = str(val)
        table.append(only_H)
    return table


def optimal_fly_param(m, H, like_array=False):
    q_km_min = []
    H_opt = []
    V_opt = []
    M_opt = []
    for i in H:
        calc = Calculation(m, il_76, i)
        result = calc.find_min_fuel_consumption()
        if result:
            q_km, mach, V = result
            q_km_min.append(q_km)
            H_opt.append(i)
            V_opt.append(V)
            M_opt.append(mach)
        else:
            break
    if like_array:
        return H_opt, V_opt, M_opt, q_km_min

    mfi = dh.get_min_or_max(q_km_min)
    return H_opt[mfi], V_opt[mfi], M_opt[mfi], q_km_min[mfi]


def cruise_fly_sim(m0, mk):
    time_tick = 1  # sec
    now_date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    log_name = f"{now_date}_sim_with_t_t_{str(time_tick)}.txt"
    H = np.arange(5000, 13000, 300)
    total_range = 0
    # begin with optimal height and speed
    H_opt, V_opt, M_opt, q_km = optimal_fly_param(m0, H)
    H_opt = 11000
    total_mass = m0
    total_time = 0
    while total_mass > mk:
        # q_km [kg/km], V [m/s]
        print(
            f"fuel_remaning= {round(total_mass - mk)} kg, total_range = {total_range}, height = {H_opt}"
        )
        q_km, _, V = min_fuel_consumption(total_mass, il_76, H_opt)
        S = V * time_tick
        S_km = S / 1000
        total_range += S_km
        fuel_burned = q_km * S_km
        total_mass -= fuel_burned

        output = f"{H_opt},{total_range},{V},{q_km},{total_mass}"
        finded_H_opt, _, _, finded_q_km_opt = optimal_fly_param(total_mass, H)
        percent_difference = ((q_km / finded_q_km_opt) - 1) * 100
        fuel_remaning = round(total_mass - mk)
        write_in_file(log_name, output)
        if percent_difference >= 1.5:
            H_opt = H_opt
    return total_range


def write_in_file(file_name, data):
    with open(file_name, "a") as f:
        f.write(str(data) + "\n")
    return True


def delete_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
    else:
        pass
    return True


il_76 = PlaneData()
m0 = il_76.MTOW
m_k = m0 - il_76.TFL
calc = Calculation(m0, il_76, 0)
Vy_speeds = calc.find_Vy_speeds()
theta_angle =np.degrees(Vy_speeds/calc.V_calc)

theta_climb = max(theta_angle)
max_index = dh.get_index_element(theta_angle, theta_climb)
tangent_line_function = dh.find_linear_func([0, calc.V_calc[max_index]], [0, Vy_speeds[max_index]])
tangent_line_x_values = [0, calc.V_calc[max_index] + 20] 
tangent_line_y_values = [tangent_line_function(value) for value in tangent_line_x_values] 

#
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('$V, \\ [м/с]$')
ax1.set_ylabel('$V_y, \\ [м/с]$', color=color)
ax1.plot(calc.V_calc, Vy_speeds, color=color)
ax1.plot(calc.V_calc[max_index], Vy_speeds[max_index], 'ro')
ax1.plot(tangent_line_x_values, tangent_line_y_values, 'g')

ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0, max(Vy_speeds)+5])
ax1.set_xlim([0, max(calc.V_calc)])

ax1.grid(which='both')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('$\\theta, \\ [град] $', color=color)  # we already handled the x-label with ax1
ax2.plot(calc.V_calc, theta_angle, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0, max(theta_angle)+ 3])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.xticks(np.arange(0, max(calc.V_calc)+1, 50))
plt.savefig('out1.png')
plt.clf()

#
#fig, ax1 = plt.subplots()
#ax1.plot(calc.V_calc, Vy_speeds, color='red', label='$V_y(V)$')
#ax2 = ax1.twinx()
#ax2.plot(calc.V_calc, theta_angle, color='blue', label='$\\theta(V)$')
#fig.tight_layout()
#
#plt.xlabel('$V, [m/s]$')
#plt.ylabel('$V_y, [m/s]$')
#plt.grid()
#plt.legend()


# print(f'start mass = {m0}, end mass = {m_k}')
# L = integrate.quad(L_range, m_k, m0)
# L = trapezoid(L_range, m_k, m0)
# L = cruise_fly_sim(m0, m_k)
# print(f'Range = {L}')

# plt.plot(mass_array, mach_optimal, label='M_{opt}')
# plt.xlabel('m, [kg]')
# plt.ylabel('MACH')
# plt.grid()
# plt.legend()
# plt.savefig('out1.png')
# plt.clf()
#
# plt.plot(mass_array, H_optimal, label='H_{opt}')
# plt.xlabel('m, [kg]')
# plt.ylabel('H, [m]')
# plt.grid()
# plt.legend()
# plt.savefig('out2.png')
# plt.clf()
#
# plt.plot(mass_array, q_km_min_optimal, label='q_{km}_{min}')
# plt.xlabel('m, [kg]')
# plt.ylabel('q_{km}_{min}')
# plt.grid()
# plt.legend()
# plt.savefig('out3.png')
# plt.clf()
