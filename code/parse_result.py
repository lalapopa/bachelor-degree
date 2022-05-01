import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as intr
from sklearn import datasets, linear_model
from statistics import mean

# matplotlib.use("pgf")
# matplotlib.rcParams.update(
#    {
#        "pgf.texsystem": "pdflatex",
#        "font.family": "serif",
#        "text.usetex": True,
#        "pgf.rcfonts": False,
#        "pgf.preamble": "\n".join(
#        [
#            r"\usepackage[T2A]{fontenc}",
#            r"\usepackage[warn]{mathtext}",
#            r"\usepackage[utf8]{inputenc}",
#            r"\usepackage[english,russian]{babel}",
#        ]
#        ),
#    }
# )


#'{H_opt},{total_range},{V},{q_km},{total_mass}'
file_name = [
    "opt_sim_trajectory_t_1.txt",
    "optimal_trajectory_t_t_60.txt",
    "psudo_optimal_trajectory.txt",
    "7500_flight_t_t_60.txt",
    "11000_flight_t_t_60.txt",
]

file_name = [
    "11_03_2022_16_20_21_sim_with_t_t_1.txt",
    "11_03_2022_16_20_02_sim_with_t_t_1.txt",
    "11_03_2022_16_19_24_sim_with_t_t_1.txt",
    "opt_sim_trajectory_t_1.txt",
]


for i, val in enumerate(file_name):
    data = pd.read_csv(val, sep=",", header=None)
    name_change = ["H", "L", "V", "q_km", "m"]
    data.columns = name_change

    H_data = np.array(data.loc[:, "H"])
    L_data = np.array(data.loc[:, "L"])
    V_data = np.array(data.loc[:, "V"])
    q_data = np.array(data.loc[:, "q_km"])
    print(f"{val}, avg = {mean(q_data)}")
    L_int = np.linspace(L_data[0], L_data[-1], 10000)
    V_int = np.linspace(V_data[0], V_data[-1], 10000)

#    model = linear_model.LinearRegression().fit(L_data.reshape((-1, 1)), V_data.reshape((-1, 1)))
#    V_regre = model.predict(L_int.reshape(-1, 1))
#
#    if i == 0:
#        model = linear_model.LinearRegression().fit(L_data.reshape((-1, 1)), V_data.reshape((-1, 1)))
#        V_regre = model.predict(L_int.reshape(-1, 1))
#        plt.plot(L_int, V_regre, label='Регрессионная модель оптимальной скорости')
#    else:
#        plt.plot(L_data, V_data, label='Псевдо оптимальное изменение скорости ')
#
# plt.legend()
##plt.plot(L_data, H_data, '--', label='Original')
# plt.grid()
# plt.xlabel('L, [km]')
# plt.ylabel('V, [м/с]')
# plt.savefig('fig1.pgf')
