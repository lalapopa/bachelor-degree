import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from statistics import mean
import os

import config
from DataHandler import DataHandler as dh


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
    return float(m0 - np.unique(np.min(m_array)))


def get_total_flight_time(time_stamp, L_array, climb_state):
    climb_time_stamp = 1
    climb_index = np.where(climb_state == 1)[0]
    climb_time = (len(climb_index)*climb_time_stamp)/60 # in minutes
    level_flight_index = np.where(climb_state == 0)[0]
    level_flight_time = (len(level_flight_index)*time_stamp)/60
    return level_flight_time + climb_time 

def split_data(L_array, data_array):
    chop_point = 3000
    index_where_L_greater_then_chops = np.where(L_array >= chop_point)[0][0]
    f = interpolate.interp1d( 
           L_array[index_where_L_greater_then_chops-1: index_where_L_greater_then_chops+1], 
           data_array[index_where_L_greater_then_chops-1: index_where_L_greater_then_chops+1],
           )
    last_data_value = f(chop_point)
    data_array = data_array[0:index_where_L_greater_then_chops+1]
    data_array[-1] = last_data_value 
    L_array = L_array[0:index_where_L_greater_then_chops+1] 
    L_array[-1] = chop_point 
    return L_array, data_array

def remove_climb_data(data, climb_state):
    mask = np.array(climb_state, dtype=bool)
    return np.delete(data, mask)


def crate_and_save_latex_table(file_name, avg_fuel_burn, flight_range, fuel_burned, time):
    text_column = np.array([
        r'$q_{км\, ср},\, \frac{кг}{км}$',
        r'$L,\, м$',
        r'$m_{сож. топл.},\, кг$',
        r'$t_{кр},\, мин$',
        ])
    value_column = np.array([
        "%.3f" % avg_fuel_burn, 
        "%.0f" % flight_range, 
        "%.2f" % fuel_burned, 
        "%.2f" % time], dtype=object)
    data = pd.DataFrame(np.array([value_column]), columns=text_column)
    latex_output = data.style.hide(axis=0)
    latex_output = dh._format_latex_table(latex_output)
    with open(file_name, 'w') as f:
        f.write(latex_output)



#'{climb or not?},{H_opt},{total_range},{V},{q_km},{total_mass}'
m0 = 180000 
t_0 = 60

file_name = os.listdir(config.PATH_SAVE_DATA)
file_name = [f for f in file_name if ".txt" in f]
plot_options = []

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

    
    L_data, H_data = split_data(L_data, H_data)
    _, climb_state = split_data(L_data, climb_state)
    _, V_data = split_data(L_data, V_data)
    _, q_data = split_data(L_data, q_data)
    _, m_data = split_data(L_data, m_data)
    total_fuel_lost = get_total_fuel_burned(m0, m_data)
    crate_and_save_latex_table(
        f"{config.PATH_TABLE}{val[0:-4]}_result.tex",
        total_fuel_lost/np.max(L_data), 
        np.max(L_data), 
        total_fuel_lost, 
        get_total_flight_time(t_0, L_data, climb_state),
    )
    print("=" * 10, val, "=" * 10)
    print(f"AVG fuel_q_km = {np.mean(q_data)}")
    print(f"Flight_range = {np.max(L_data)}")
    print(
        f"Total_fuel burned = {get_total_fuel_burned(m0, m_data)}\n"
        f"Total Flight time = {get_total_flight_time(t_0, L_data, climb_state)}"
    )
    H_data = remove_climb_data(H_data, climb_state)
    L_data = remove_climb_data(L_data, climb_state)
    V_data = remove_climb_data(V_data, climb_state)
    q_data = remove_climb_data(q_data, climb_state)
    m_data = remove_climb_data(m_data, climb_state)

    if i == 0 or i == 2:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        polynomial_features_H = PolynomialFeatures(degree=2, include_bias=False)
        linear_regression_H = LinearRegression()
        pipeline_H = Pipeline(
            [
                ("polynomial_features", polynomial_features_H),
                ("linear_regression", linear_regression_H),
            ]
        )
        pipeline_H.fit(L_data[:, np.newaxis], H_data)
        if i == 2:
            polynomial_features_V = PolynomialFeatures(degree=1, include_bias=False)
        else:
            polynomial_features_V = PolynomialFeatures(degree=2, include_bias=False)
        linear_regression_V = LinearRegression()
        pipeline_V = Pipeline(
            [
                ("polynomial_features", polynomial_features_V),
                ("linear_regression", linear_regression_V),
            ]
        )
        pipeline_V.fit(L_data[:, np.newaxis], V_data)
        L_data_int = np.linspace(L_data[0], L_data[-1], 100)

        ax1.plot(L_data_int, pipeline_H.predict(L_data_int[:, np.newaxis]), "g", label="$H(L)$")
        ax2.plot(L_data_int, pipeline_V.predict(L_data_int[:, np.newaxis]), "b", label="$V(L)$")
        ax2.set_ylim([min(V_data)-15, max(V_data) + 5])

        ax1.set_xlabel("L, [km]")
        ax1.set_ylabel("H, м", color="g")
        ax2.set_ylabel("V, м/с", color="b")
        fig.legend(loc=3, frameon=True, bbox_to_anchor=(0.75, 0.10))
        ax1.grid()
        plt.savefig(f"{config.PATH_FIGURES}{val[0:-4]}_L_H.pgf")
        plt.clf()

    else:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(L_data, H_data, "g", label="$H(L)$")
        ax2.plot(L_data, V_data, "b", label="$V(L)$")
        ax2.set_ylim([min(V_data)-15, max(V_data) + 5])

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
    ax2.set_ylim([min(q_data)-1, max(q_data)+1])
    ax1.set_ylim([100000, max(m_data) + 5000])
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
