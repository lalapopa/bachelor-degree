import os
import scipy.integrate as integrate
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ambiance import Atmosphere
from matplotlib import rc
from datetime import datetime
import multiprocessing
import concurrent.futures
from functools import partial

from lerp import linear1d
from DataHandler import DataHandler as dh
from plane_data import PlaneData
from aerodynamics_data import AerodynamicsData
from formulas import Formulas as eq
import config


def pgf_setting():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage[warn]{mathtext}",
                    r"\usepackage[T2A]{fontenc}",
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[english,russian]{babel}",
                ]
            ),
        }
    )


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
        otn_P_0 =  eq.thrust_to_weight_equation(
                self.plane_char.zero_thrust_one_eng,
                self.plane_char.N_DV,
                self.mass,
                self.g,
                )
        self.P_rasp = eq.P_rasp_equation(
            otn_P_0,
            self.mass,
            self.g,
            tilda_P,
            self.H,
            air_11.pressure[0],
            air_H.pressure[0],
        )
        print(f'if H = {H}, MACH = {round(self.MACH_calc[300],2)} K= {round(K[300],2)}, Cy = {round(Cy[300], 3)}, T/W ration = {otn_P_0}')

    def find_min_fuel_consumption(self):
        P_diff = self.P_rasp - self.P_potr
        if self._can_fly(P_diff):
            tilda_R = eq.tilda_R_equation(self.P_potr, self.P_rasp)
            Ce_tilda = [self.ad.CeTilda(M, self.H) for M in self.MACH_calc]
            Ce_dr = self.ad.Ce_dr_value(tilda_R)
            #            print(f'Tilda R = {min(tilda_R)}, Ce_dr = {min(Ce_dr)}')
            #            print(f'index = {np.where(tilda_R == np.min(tilda_R))}')
#            print(f"{self.V_calc[np.where(tilda_R == np.min(tilda_R))[0]]}")

            Ce = eq.Ce_equation(self.plane_char.CE_0, Ce_tilda, Ce_dr)
            q_chas = eq.q_ch_hour_consumption(Ce, self.P_potr)

            self.q_km = eq.q_km_range_consumption(q_chas, self.V_calc)
#            print(
#                f"min q km = { np.min(self.q_km) }, V min q  = {self.V_calc[np.where(self.q_km == np.min(self.q_km))]}"
#            )

            min_km_fuel_index = dh.get_min_or_max(self.q_km)
            mach_min_fuel = self.V_calc[min_km_fuel_index] / self.a_sos

            return (
                self.q_km[min_km_fuel_index],
                mach_min_fuel,
                self.V_calc[min_km_fuel_index],
            )
        #        return q_chas[min_km_fuel_index], mach_min_fuel, V_calc[min_km_fuel_index]
        else:
            return (False, False, False)

    def _can_fly(self, thrust_diff):
        for i in thrust_diff:
            if np.sign(i) > 0:
                return True
        return False

    def find_Vy_speeds(self):
        return self.V_calc * (self.P_rasp - self.P_potr) / (self.mass * self.g)


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


def divide_into_chunks(array):
    cpu_number = multiprocessing.cpu_count()
    img_size = list(array.shape)
    divide = int(img_size[0] / cpu_number)
    chunks = []
    if img_size[0] > cpu_number:
        for val in range(0, cpu_number):
            right_limit = divide * (val + 1)
            left_limit = val * divide
            if val == cpu_number - 1:
                right_limit = img_size[0]
            chunks.append(array[left_limit:right_limit])
    else:
        chunks = [array]
    return chunks


def paralell_optimal_fly_param(H, mass):
    if type(H) == np.int64:
        chunks = H
        H_opt, V_opt, M_opt, q_km_min = optimal_fly_param(mass, chunks)
        return (
            H_opt,
            V_opt,
            M_opt,
            q_km_min,
        )
    else:
        chunks = divide_into_chunks(H)
    init_function = partial(optimal_fly_param, mass)
    H_opt_array = []
    V_opt_array = []
    M_opt_array = []
    q_km_min_array = []
    with concurrent.futures.ProcessPoolExecutor() as exe:
        result = exe.map(init_function, chunks)
        for H_opt, V_opt, M_opt, q_km_min in result:
            H_opt_array.append(H_opt)
            V_opt_array.append(V_opt)
            M_opt_array.append(M_opt)
            q_km_min_array.append(q_km_min)
    min_q_km_index = np.unique(np.where(q_km_min_array == np.min(q_km_min_array)))[0]
    print("FUEL CONSUMPTION = ", q_km_min_array[min_q_km_index])
    return (
        H_opt_array[min_q_km_index],
        V_opt_array[min_q_km_index],
        M_opt_array[min_q_km_index],
        q_km_min_array[min_q_km_index],
    )


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

    if not q_km_min:
        return (999, 999, 999, 999)

    mfi = np.unique(np.where(q_km_min == np.min(q_km_min)))[0]
    return H_opt[mfi], V_opt[mfi], M_opt[mfi], q_km_min[mfi]


def cruise_fly_sim(m0, mk, L_k):
    time_tick = 60  # sec
    now_date = datetime.now().strftime("%Y%m%d%H%M%S")
    log_name = f"{now_date}_sim_with_t_t_{str(time_tick)}.txt"
    H = np.arange(9500, 12000, 10)
    #    H = np.array([10000])
    total_range = 0
    # begin with optimal height and speed
    H_opt, _, _, _ = paralell_optimal_fly_param(H, m0)
    total_mass = m0
    while total_mass > mk and total_range < L_k:
        # q_km [kg/km], V [m/s]
        calc = Calculation(total_mass, il_76, H_opt)
        q_km, _, V = calc.find_min_fuel_consumption()
        S = V * time_tick
        S_km = S / 1000
        total_range += S_km
        fuel_burned = q_km * S_km
        total_mass -= fuel_burned

        output = f"{H_opt},{total_range},{V},{q_km},{total_mass}"
        write_in_file(log_name, output)

        finded_H_opt, _, _, finded_q_km_opt = paralell_optimal_fly_param(H, total_mass)
        H_diff = finded_H_opt - H_opt
        fuel_remaning = total_mass - mk

        if H_diff >= 10:
            L_array, H_array, V_array, q_array, mass_change_array = calulate_climb(
                H_opt, finded_H_opt, total_mass, il_76
            )
            H_opt = finded_H_opt
            for i, value in enumerate(L_array):
                output = f"{H_array[i]},{total_range+(L_array[i]/1000)},{V_array[i]},{q_array[i]},{mass_change_array[i]}"
                write_in_file(log_name, output)
            total_mass = mass_change_array[-1]
            total_range += L_array[-1] / 1000
        print(
            f"fuel_remaning= {fuel_remaning} kg, total_range = {total_range}, height = {H_opt}, speed = {V}"
        )
    return total_range


def write_in_file(file_name, data):
    with open(config.PATH_DATA + file_name, "a") as f:
        f.write(str(data) + "\n")
    return True


def delete_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
    else:
        pass
    return True


def calulate_climb(H_0, H_k, m0, plane):
    time_stamp = 1
    H_current = H_0
    L = 0
    L_array = []
    H_array = []
    V_array = []
    q_array = []
    mass_change_array = []
    m_climb = m0
    while H_current < H_k:
        print(f"Climbint to {H_k}/{H_current} m | Traveled dist {L}")

        calc = Calculation(m_climb, il_76, H_current)
        Vy_speeds = calc.find_Vy_speeds()
        _, _, _ = calc.find_min_fuel_consumption()

        theta_angle = np.degrees(Vy_speeds / calc.V_calc)
        Vy_max = np.max(Vy_speeds)

        index_max_Vy = np.unique(np.where(Vy_speeds == Vy_max))[0]
        V_climb = calc.V_calc[index_max_Vy]
        q_km_climb = calc.q_km[index_max_Vy]

        S_v = Vy_max * time_stamp
        S_h = V_climb * time_stamp
        S_h_km = S_h / 1000

        fuel_burned = q_km_climb * S_h_km
        m_climb -= fuel_burned

        H_current += S_v
        L += S_h

        H_array.append(H_current)
        L_array.append(L)
        V_array.append(V_climb)
        q_array.append(q_km_climb)
        mass_change_array.append(m_climb)
        print(f"{H_current},{L},{V_climb},{q_km_climb},{m_climb}")
        print(f"V_y = {Vy_max}")

    return L_array, H_array, V_array, q_array, mass_change_array

#pgf_setting()
il_76 = PlaneData()
#m0 = il_76.MTOW
#m_k = m0 - il_76.TFL - 10000
#L_k = 4000

# print(f'start mass = {m0}, end mass = {m_k}')
##L = integrate.quad(L_range, m_k, m0)
##L = trapezoid(L_range, m_k, m0)
# L = cruise_fly_sim(m0, m_k, L_k)
# print(f'Range = {L}')

# print(m_k)
# calc = Calculation(140000, il_76, 10000)
# Vy_speeds = calc.find_Vy_speeds()
# q_km, mach_min, V_min = calc.find_min_fuel_consumption()
# print(max(Vy_speeds))
# print(V_min)

mass = [170000, 150000, 140000]
H_calc = 9000 

for i in mass:
    calc = Calculation(i, il_76, H_calc)
    Vy_speeds = calc.find_Vy_speeds()
    q_km, mach_min, V_min = calc.find_min_fuel_consumption()
    V_calc = calc.V_calc
    q_km_calc = calc.q_km
    index_min_fuel = np.where(q_km_calc == np.min(q_km_calc))
    label_q_min = "$q_{{км}_{min}}=%.3f$" % (q_km)
    text_V = "$V = %.2f$" % (V_calc[index_min_fuel])
    title_text = "Для высоты H= %.0f м, m = %.0f кг" % (H_calc, i)

    plt.plot(
        V_calc[index_min_fuel],
        q_km_calc[index_min_fuel],
        "o",
        color="r",
        label=label_q_min,
    )
    plt.text(V_calc[index_min_fuel], q_km_calc[index_min_fuel] + 2, text_V)
    plt.title(title_text)

    plt.plot(V_calc, q_km_calc, label="$q_{km}(V)$")
    plt.xlabel(r"$V, \frac{м}{с}$")
    plt.ylabel(r"$q_{km}, \frac{кг}{км}$")

    plt.grid()
    plt.legend()
    plt.savefig(f"{i}_q_km_V.pgf")
    plt.clf()

# L, H, V_array, q_km, mass_change  = calulate_climb(9000, 13000, 166000, il_76)
#
# plt.plot(calc.MACH_calc, calc.q_km)
# plt.xlabel('Mach')
# plt.ylabel('q')
# plt.grid()
# plt.legend()
#
# plt.savefig('out1.png')
# plt.clf()
#

# calc = Calculation(m0, il_76, 10)
# Vy_speeds = calc.find_Vy_speeds()
##print(Vy_speeds)
# theta_angle = np.degrees(Vy_speeds/calc.V_calc)
# Vy_max = np.max(Vy_speeds)
# theta_climb = theta_angle[np.unique(np.where(Vy_speeds == Vy_max))[0]]


# max_index = dh.get_index_element(theta_angle, theta_climb)
# tangent_line_function = dh.find_linear_func([0, calc.V_calc[max_index]], [0, Vy_speeds[max_index]])
# tangent_line_x_values = [0, calc.V_calc[max_index] + 20]
# tangent_line_y_values = [tangent_line_function(value) for value in tangent_line_x_values]
#
#
# fig, ax1 = plt.subplots()
# color = '#5e3c99'
# ax1.set_xlabel('$V, \\ [м/с]$')
# ax1.set_ylabel('$V_y, \\ [м/с]$', color=color)
# ax1.plot(calc.V_calc, Vy_speeds, color=color)
# ax1.plot(calc.V_calc[max_index], Vy_speeds[max_index], 'ro')
# ax1.plot(tangent_line_x_values, tangent_line_y_values, 'g')
#
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_ylim([0, max(Vy_speeds)+5])
# ax1.set_xlim([0, max(calc.V_calc)])
#
# ax1.grid(which='both')
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = '#e66101'
# ax2.set_ylabel('$\\theta, \\ [град] $', color=color)  # we already handled the x-label with ax1
# ax2.plot(calc.V_calc, theta_angle, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_ylim([0, max(theta_angle)+ 3])
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.xticks(np.arange(0, max(calc.V_calc)+1, 50))
# plt.savefig('climbing_angle.png')
# plt.clf()


# fig, ax1 = plt.subplots()
# ax1.plot(calc.V_calc, Vy_speeds, color='red', label='$V_y(V)$')
# ax2 = ax1.twinx()
# ax2.plot(calc.V_calc, theta_angle, color='blue', label='$\\theta(V)$')
# fig.tight_layout()
#
# plt.xlabel('$V, [m/s]$')
# plt.ylabel('$V_y, [m/s]$')
# plt.grid()
# plt.legend()
#
# plt.plot(mass_array, mach_optimal, label='M_{opt}')
# plt.xlabel('m, [kg]')
# plt.ylabel('MACH')
# plt.grid()
# plt.legend()
# plt.savefig('out1.png')
# plt.clf()
#
#  plt.plot(mass_array, H_optimal, label='H_{opt}')
# p plt.xlabel('m, [kg]')
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
