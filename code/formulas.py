import numpy as np

class Formulas:
    def q_dynamic_pressure(v, air_density):
        return (air_density * (v ** 2)) / 2

    def V_speed(mach, sos):
        return mach * sos

    def C_y_n_lift_coefficient(otn_M, ps, q):
        return (otn_M * ps * 10) / q

    def my_C_y_n(m, g, S, ro, M, a_sos):
        return (2 * m * g) / (S * ro * np.power(M, 2) * np.power(a_sos, 2))

    def C_x_n_drag_coefficient(C_x_m, A, C_y_n, C_y_m):
        return C_x_m + A * (C_y_n - C_y_m) ** 2

    def K_n_lift_to_drag_ratio(C_y_n, C_x_n):
        return np.nan_to_num(C_y_n / C_x_n)

    def P_potr_equation(M0, g, K_n):
        return (M0 * g) / K_n

    def P_rasp_equation(otn_P_0, M0, g, tilda_P, alt, p_h_11, p_h):
        if alt >= 11000:
            return otn_P_0 * M0 * g * tilda_P * (p_h / p_h_11)
        else:
            return otn_P_0 * M0 * g * tilda_P

    def q_ch_hour_consumption(Ce, P_potr):
        return Ce * P_potr

    def q_km_range_consumption(q_ch, V):
        return q_ch / (3.6 * V)

    def tilda_R_equation(P_potr, P_rasp):
        return P_potr / P_rasp

    def Ce_equation(Ce_0, tilda_Ce, tilda_Ce_dr):
        tilda_Ce = np.array(tilda_Ce)
        tilda_Ce_dr = np.array(tilda_Ce_dr)
        return Ce_0 * tilda_Ce * tilda_Ce_dr

    def sin_theta_nab(P_rasp, P_potr, m, g):
        return (P_rasp - P_potr) / (m * g)

    def V_y_eq(V, sin_theta):
        return V * sin_theta
