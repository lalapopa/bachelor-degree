import os 
import scipy.integrate as integrate 
import numpy as np 
import matplotlib.pyplot as plt 
from ambiance import Atmosphere
from matplotlib import rc
import json
from datetime import datetime

from my_integral_calc import trapezoid
from lerp import linear1d 
from plane_data import PlaneData
from aero_data import AeroData
from data_handler import DataHandler as dh


def get_Vy_value(mass, plane_char, H):
    H = float(H) # ambiance package can't take np.int16 type 
    air_H = Atmosphere(H)
    air_11 = Atmosphere(11000)
    g = 9.81
    Ro = air_H.density[0]
    a_sos = air_H.speed_of_sound[0]

    MACH_calc = np.arange(0.3, 0.95, 0.001) 
    V_calc = V_speed(MACH_calc, a_sos)
    q = q_dynamic_pressure(V_calc, Ro)

    Cy = my_C_y_n(mass, g, plane_char.WING_AREA,Ro, MACH_calc, a_sos) 
    A = aero.A_value(MACH_calc) 
    Cxm = aero.Cxm_value(MACH_calc)
    Cym = aero.Cym_value(MACH_calc)
    Cx = C_x_n_drag_coefficient(Cxm, A, Cy, Cym)

    K = K_n_lift_to_drag_ratio(Cy, Cx)

    P_potr = P_potr_equation(mass, g, K)
    tilda_P = np.array([aero.PTilda(M, H) for M in MACH_calc])
    P_rasp = P_rasp_equation(
    plane_char.OTN_P_0,
    mass,
    g,
    tilda_P,
    H,
    air_11.pressure[0],
    air_H.pressure[0]
    )









