import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as intr
from sklearn import datasets, linear_model
from statistics import mean
import os

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


def get_total_fuel_burned(m0, m_array):
    return m0 - np.unique(np.min(m_array))


def get_total_flight_time(time_stamp, L_array, climb_state):
    climb_time_stamp = 1
    climb_index = np.where(climb_state == 1)[0]
    climb_time = (len(climb_index)*climb_time_stamp)/60 # in minutes
    level_flight_index = np.where(climb_state == 0)[0]
    level_flight_time = (len(level_flight_index)*time_stamp)/60
    return level_flight_time + climb_time 


#'{H_opt},{total_range},{V},{q_km},{total_mass}'
m0 = 166000
t_0 = 60

file_name = os.listdir(config.PATH_SAVE_DATA)
file_name = [f for f in file_name if ".txt" in f][0:]

for i, val in enumerate(file_name):
    data = pd.read_csv(config.PATH_SAVE_DATA + val, sep=",", header=None)
    name_change = ["climb_state", "H", "L", "V", "q_km", "m"]
    data.columns = name_change
    
    climb_state = np.array(data.loc[:, "climb_state"])
    H_data = np.array(data.loc[:, "H"])
    L_data = np.array(data.loc[:, "L"])
    V_data = np.array(data.loc[:, "V"])
    q_data = np.array(data.loc[:, "q_km"])
    m_data = np.array(data.loc[:, "m"])

    print("=" * 10, val, "=" * 10)
    print(f"AVG fuel_q_km = {np.average(q_data)}")
    print(f"Flight_range = {np.max(L_data)}")
    print(
        f"Total_fuel burned = {get_total_fuel_burned(m0, m_data)}\nTotal Flight time = {get_total_flight_time(t_0, L_data, climb_state)}"
    )

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(L_data, H_data, "g", label="$H(L)$")
    ax2.plot(L_data, V_data, "b", label="$V(L)$")
    ax2.set_ylim([100, max(V_data) + 2])

    ax1.set_xlabel("L, [km]")
    ax1.set_ylabel("H, м", color="g")
    ax2.set_ylabel("V, м/с", color="b")

    fig.legend(loc=3, frameon=True, bbox_to_anchor=(0.75, 0.10))
    ax1.grid()
    plt.savefig(f"{config.PATH_FIGURES}{val[0:-4]}_L_H.pgf")
    plt.clf()

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(L_data, m_data, "g", label="$m(L)$")
    ax2.plot(L_data, q_data, "b", label="$q(L)$")

    ax1.set_xlabel("L, [km]")
    ax1.set_ylabel("m, [кг]", color="g")
    ax2.set_ylabel("q_km, [kg/km]", color="b")

    fig.legend(loc=3, frameon=True, bbox_to_anchor=(0.75, 0.10))
    ax1.grid()
    plt.savefig(f"{config.PATH_FIGURES}{val[0:-4]}_L_m.pgf")
    plt.clf()


#    L_int = np.linspace(L_data[0], L_data[-1], 10000)
#    V_int = np.linspace(V_data[0], V_data[-1], 10000)
#
#    model = linear_model.LinearRegression().fit(L_data.reshape((-1, 1)), V_data.reshape((-1, 1)))
#    V_regre = model.predict(L_int.reshape(-1, 1))
#
#    if i == 0:
#        model = linear_model.LinearRegression().fit(L_data.reshape((-1, 1)), V_data.reshape((-1, 1)))
#        V_regre = model.predict(L_int.reshape(-1, 1))
#        plt.plot(L_int, V_regre, label='Регрессионная модель оптимальной скорости')
#    else:
#    plt.plot(L_data, H_data)
##    plt.legend()
##plt.plot(L_data, H_data, '--', label='Original')
#    plt.grid()
#    plt.xlabel('L, [km]')
#    plt.ylabel('H, [м/с]')
#    plt.savefig(f'{config.PATH_FIGURES}{val[0:-4]}.png')
#    plt.clf()
#
#    plt.plot(L_data, V_data)
##    plt.legend()
##plt.plot(L_data, H_data, '--', label='Original')
#    plt.grid()
#    plt.xlabel('L, [km]')
#    plt.ylabel('V, [м/с]')
#    plt.savefig(f'{config.PATH_FIGURES}{val[0:-4]}_V.png')
#    plt.clf()
#
