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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

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
        self.MACH_calc = np.arange(0.3, 0.95, 0.1)
        self.g = 9.81

        self.mass = mass
        self.H = float(H)  # ambiance package can't take np.int16 type
        air_H = Atmosphere(self.H)
        air_11 = Atmosphere(11000)
        Ro = air_H.density[0]
        self.a_sos = air_H.speed_of_sound[0]

        self.V_calc = eq.V_speed(self.MACH_calc, self.a_sos)
        q = eq.q_dynamic_pressure(self.V_calc, Ro)

        self.Cy = eq.my_C_y_n(
            self.mass, self.g, self.plane_char.WING_AREA, Ro, self.MACH_calc, self.a_sos
        )
        A = self.ad.A_value(self.MACH_calc)
        Cxm = self.ad.Cxm_value(self.MACH_calc)
        Cym = self.ad.Cym_value(self.MACH_calc)
        self.Cx = eq.C_x_n_drag_coefficient(Cxm, A, self.Cy, Cym)
        #        print(
        #            f"Cym = {Cym[300]}, Cxm = {Cxm[300]}, Cx = {self.Cx[300]}, Cy = {self.Cy[300]}"
        #        )
        K = eq.K_n_lift_to_drag_ratio(self.Cy, self.Cx)

        self.P_potr = eq.P_potr_equation(self.mass, self.g, K)
        tilda_P = np.array([self.ad.PTilda(M, self.H) for M in self.MACH_calc])
        self.K = K
        otn_P_0 = eq.thrust_to_weight_equation(
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

    #        print(f'if H = {H}, MACH = {round(self.MACH_calc[300],2)} K= {round(K[300],2)}, self.Cy = {round(self.Cy[300], 3)}, T/W ration = {otn_P_0}')

    def find_min_fuel_consumption(self):
        P_diff = self.P_rasp - self.P_potr
        if self._can_fly(P_diff):
            if self.H < 11000:
                tilda_R = eq.tilda_R_equation(self.P_potr, self.P_rasp)
                Ce_tilda = [self.ad.CeTilda(M, self.H) for M in self.MACH_calc]
                Ce_dr = self.ad.Ce_dr_value(self.H, tilda_R)
                self.Ce = eq.Ce_equation(self.plane_char.CE_0, Ce_tilda, Ce_dr)
            else:
                self.Ce = self._ce_11_km()
            self.V_int = np.linspace(self.V_calc[0], self.V_calc[-1], 250)
            self.P_potr_int = Calculation.approximate_like_polynom(
                self.P_potr, self.V_calc, self.V_int
            )
            self.Ce_int = Calculation.approximate_like_polynom(
                self.Ce, self.V_calc, self.V_int
            )

            q_chas = eq.q_ch_hour_consumption(self.Ce_int, self.P_potr_int)
            self.q_km = eq.q_km_range_consumption(q_chas, self.V_int)

            #            print(
            #                f"min q km = { np.min(self.q_km) }, V min q  = {self.V_int[np.where(self.q_km == np.min(self.q_km))]}"
            #            )
            min_km_fuel_index = dh.get_min_or_max(self.q_km)
            mach_min_fuel = self.V_int[min_km_fuel_index] / self.a_sos

            return (
                self.q_km[min_km_fuel_index],
                mach_min_fuel,
                self.V_int[min_km_fuel_index],
            )
        #        return q_chas[min_km_fuel_index], mach_min_fuel, V_calc[min_km_fuel_index]
        else:
            return (999, 999, 999)

    def _ce_11_km(self):
        air_H = Atmosphere(self.H)
        air_11 = Atmosphere(11000)
        Ro = air_H.density[0]
        a_sos = air_H.speed_of_sound[0]
        V_calc = eq.V_speed(self.MACH_calc, a_sos)
        q = eq.q_dynamic_pressure(V_calc, Ro)

        Cy = eq.my_C_y_n(
            self.mass, self.g, self.plane_char.WING_AREA, Ro, self.MACH_calc, a_sos
        )
        A = self.ad.A_value(self.MACH_calc)
        Cxm = self.ad.Cxm_value(self.MACH_calc)
        Cym = self.ad.Cym_value(self.MACH_calc)
        Cx = eq.C_x_n_drag_coefficient(Cxm, A, Cy, Cym)
        K = eq.K_n_lift_to_drag_ratio(Cy, Cx)

        P_potr = eq.P_potr_equation(self.mass, self.g, K)
        tilda_P = np.array([self.ad.PTilda(M, 11000) for M in self.MACH_calc])
        otn_P_0 = eq.thrust_to_weight_equation(
            self.plane_char.zero_thrust_one_eng,
            self.plane_char.N_DV,
            self.mass,
            self.g,
        )
        P_rasp = eq.P_rasp_equation(
            otn_P_0,
            self.mass,
            self.g,
            tilda_P,
            11000,
            air_11.pressure[0],
            air_H.pressure[0],
        )
        tilda_R = eq.tilda_R_equation(P_potr, P_rasp)
        Ce_tilda = [self.ad.CeTilda(M, 11000) for M in self.MACH_calc]
        Ce_dr = self.ad.Ce_dr_value(self.H, tilda_R)
        Ce = eq.Ce_equation(self.plane_char.CE_0, Ce_tilda, Ce_dr)
        return Ce

    def _can_fly(self, thrust_diff):
        for i in thrust_diff:
            if np.sign(i) > 0:
                return True
        return False

    @staticmethod
    def approximate_like_polynom(target_value, x_value, x_value_int, degree=3):
        polynomial_features_target = PolynomialFeatures(
            degree=degree, include_bias=False
        )
        linear_regression = LinearRegression()
        pipeline = Pipeline(
            [
                ("polynomial_features", polynomial_features_target),
                ("linear_regression", linear_regression),
            ]
        )
        pipeline.fit(x_value[:, np.newaxis], target_value)
        return pipeline.predict(x_value_int[:, np.newaxis])

    def find_Vy_speeds(self):
        P_rasp_int = Calculation.approximate_like_polynom(
            self.P_rasp, self.V_calc, self.V_int
        )
        return self.V_int * (P_rasp_int - self.P_potr_int) / (self.mass * self.g)


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
    H = np.array(H)
    H_hi_res = np.arange(H[0], H[-1] + 10, 10)
    for i in H_hi_res:
        calc = Calculation(m, il_76, i)
        q_km, mach, V = calc.find_min_fuel_consumption()
        if q_km < 999:
            q_km_min.append(q_km)
            H_opt.append(i)
            V_opt.append(V)
            M_opt.append(mach)
        else:
            break

    if len(q_km_min) == 0:
        return (999, 999, 999, 999)
    plt.plot(H_opt, q_km_min, "--", label=f"mass={m}")
    q_km_min = Calculation.approximate_like_polynom(
        np.array(q_km_min), np.array(H_opt), H, degree=3
    )
    M_opt = Calculation.approximate_like_polynom(
        np.array(M_opt), np.array(H_opt), H, degree=3
    )
    V_opt = Calculation.approximate_like_polynom(
        np.array(V_opt), np.array(H_opt), H, degree=3
    )

    plt.plot(H, q_km_min, label=f"mass={m}")

    for i, alt in enumerate(H):
        if alt in H_opt:
            continue
        q_km_min[i] = 999
        M_opt[i] = 999
        V_opt[i] = 999
    if like_array:
        return H, V_opt, M_opt, q_km_min
    mfi = np.unique(np.where(q_km_min == np.min(q_km_min)))[0]
    return H[mfi], V_opt[mfi], M_opt[mfi], q_km_min[mfi]


def cruise_fly_sim(m0, mk, L_k):
    time_tick = 60  # sec
    now_date = datetime.now().strftime("%Y%m%d%H%M%S")
    log_name = f"{now_date}_sim_with_t_t_{str(time_tick)}.txt"
    H = np.arange(9000, 13000, 10)
    total_range = 0
    # begin with optimal height and speed
    print(f"Hmin {min(H)}, Hmax {max(H)}")
    H_opt, _, _, _ = optimal_fly_param(m0, H)
    H_opt = float(H_opt)
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

        output = f"0,{H_opt},{total_range},{V},{q_km},{total_mass}"
        write_in_file(log_name, output)

        finded_H_opt, _, _, finded_q_km_opt = optimal_fly_param(total_mass, H)
        H_diff = finded_H_opt - H_opt
        fuel_remaning = total_mass - mk
        print(finded_H_opt)

        if H_diff >= 10:
            L_array, H_array, V_array, q_array, mass_change_array = calulate_climb(
                H_opt, finded_H_opt, total_mass, il_76
            )
            H_opt = finded_H_opt
            for i, value in enumerate(L_array):
                output = f"1,{H_array[i]},{total_range+(L_array[i]/1000)},{V_array[i]},{q_array[i]},{mass_change_array[i]}"
                write_in_file(log_name, output)
            total_mass = mass_change_array[-1]
            total_range += L_array[-1] / 1000
        print(
            f"fuel_remaning= {fuel_remaning} kg, total_range = {total_range}, height = {H_opt}, speed = {V}"
        )
    return total_range


def write_in_file(file_name, data):
    with open(config.PATH_SAVE_DATA + file_name, "a") as f:
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
        _, _, V_calc_q_min = calc.find_min_fuel_consumption()
        Vy_speeds = calc.find_Vy_speeds()

        index_max_Vy = np.unique(np.where(calc.V_int == V_calc_q_min))[0]
        Vy_max = Vy_speeds[index_max_Vy]
        V_climb = calc.V_int[index_max_Vy]
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
    return L_array, H_array, V_array, q_array, mass_change_array


# pgf_setting()
il_76 = PlaneData()
m0 = il_76.MTOW
m_k = m0 - il_76.TFL
L_k = 4000

if __name__ == "__main__":
    print(f"start mass = {m0}, end mass = {m_k}")
    # L = integrate.quad(L_range, m_k, m0)
    # L = trapezoid(L_range, m_k, m0)
    L = cruise_fly_sim(m0, m_k, L_k)
    print(f"Range = {L}")


######################
#  Climb simulaiton  #
######################

#    L, H, V_array, q_km, mass_change = calulate_climb(8500, 9000, m0, il_76)
#    plt.plot(L, V_array)
#    plt.xlabel("L")
#    plt.ylabel("V")
#    plt.grid()
#    plt.legend()
#
#    plt.savefig("out1.png")
#    plt.clf()
#
#    plt.plot(L, H)
#    plt.xlabel("L")
#    plt.ylabel("H")
#    plt.grid()
#    plt.legend()
#
#    plt.savefig("out2.png")
#    plt.clf()
#
#    print(
#        f"main param\nTravel distance: {L[-1]/1000} km\n",
#        f"Fuel lost: {m0-mass_change[-1]}\n",
#        f"Avg speed: {np.average(V_array)}\n",
#        f"q_km cons: {(m0-mass_change[-1])/(L[-1]/1000)}",
#    )
