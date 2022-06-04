import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import re


def take_H_from_columns_name(column_names):
    H_array = []
    for name in column_names:
        try:
            H = float(re.findall("(?<=\=)[\d]*", name)[0])
        except IndexError:
            continue
        if H in H_array:
            continue
        else:
            H_array.append(H)
    return H_array


def column_pos_with_name(name, columns_name):
    pos = []
    for i, c_n in enumerate(columns_name):
        if name in c_n:
            pos.append(i)
    return pos


def interpolate_H(M, P, H, H_int):
    for i, val in enumerate(H):
        if val > H_int:
            index_r = i
            index_l = i - 1
            break
        elif val == H_int:
            return M[i], P[i]
    try:
        index_r
    except NameError:
        raise IndexError(f"{H_int} value not in {H} array")

    M_int = []
    P_int = []
    P_up = P[index_r]
    P_down = P[index_l]
    M_up = M[index_r]
    M_down = M[index_l]
    H_up = H[index_r]
    H_down = H[index_l]
    for i in range(0, len(P_up)):
        P_f = interpolate.interp1d(
            [H_up, H_down], [P_up[i], P_down[i]], fill_value="extrapolate"
        )
        P_int.append(P_f(H_int))
        M_f = interpolate.interp1d(
            [H_up, H_down], [M_up[i], M_down[i]], fill_value="extrapolate"
        )
        M_int.append(M_f(H_int))
    return M_int, P_int


def create_dataframe(columns_array, columns_names):
    df = pd.DataFrame(columns_array).T
    df = df.set_axis(column_names, axis=1, inplace=False)
    return df


save_path = "../data/"

#############
#  P_Tilde  #
#############


file_name = "P_M.csv"
df = pd.read_csv(file_name)
column_names = list(df.columns)

H_values = take_H_from_columns_name(column_names)
M_pos = column_pos_with_name("M", column_names)
P_pos = [val + 1 for val in M_pos]

M_values = df.iloc[:, M_pos].T.to_numpy()
P_values = df.iloc[:, P_pos].T.to_numpy()

MACH_int = np.arange(0, 1 + 0.05, 0.05)
H_int = np.arange(0, 11 + 1, 1)

P_array = []
column_names = []
P_array.append(MACH_int)
column_names.append("M_H_ptilda")
for i, H in enumerate(H_int):
    column_names.append("Ptilda%.0f" % H)
    M_H_values, P_H_values = interpolate_H(M_values, P_values, H_values, H)
    P_f = interpolate.interp1d(M_H_values, P_H_values, fill_value="extrapolate")
    P_int = P_f(MACH_int)
    P_H_array = []
    for P_value in P_int:
        P_tilde = P_value * 1000 / 94700
        P_H_array.append(P_tilde)
    P_array.append(P_H_array)

df = create_dataframe(P_array, column_names)
df.to_csv(save_path + "ad_data_P_H_M.csv", index=False)

#############
#  CeTilde  #
#############

file_name = "Cud_M_correct.csv"
df = pd.read_csv(file_name)
column_names = list(df.columns)

H_values = take_H_from_columns_name(column_names)
M_pos = column_pos_with_name("M", column_names)
C_pos = [val + 1 for val in M_pos]

M_values = df.iloc[:, M_pos].T.to_numpy()
C_values = df.iloc[:, C_pos].T.to_numpy()

MACH_int = np.arange(0.1, 1 + 0.05, 0.05)
H_int = np.arange(0, 11 + 1, 1)

C_array = []
column_names = []
C_array.append(MACH_int)
column_names.append("M_H_ce")

for i, H in enumerate(H_int):
    column_names.append("Cetilda%.0f" % H)
    M_H_values, C_H_values = interpolate_H(M_values, C_values, H_values, H)
    C_f = interpolate.interp1d(M_H_values, C_H_values, fill_value="extrapolate")
    C_int = C_f(MACH_int)
    C_H_array = []
    for C_value in C_int:
        C_tilde = C_value / 0.0457
        C_H_array.append(C_tilde)
    C_array.append(C_H_array)

df = create_dataframe(C_array, column_names)
df.to_csv(save_path + "ad_data_Ce_H_M.csv", index=False)

##############
#  Cy_alpha  #
##############

file_name = "Cy_alpha_grad.csv"
df = pd.read_csv(file_name)


M_values = df.iloc[:, 0].T.to_numpy()
Cy_values = df.iloc[:, 1].T.to_numpy()
print(Cy_values)
print(M_values)

MACH_int = np.arange(0.1, 1 + 0.05, 0.05)

Cy_f = interpolate.interp1d(M_values, Cy_values, fill_value="extrapolate")
Cy_int = Cy_f(MACH_int)
[print(f"MACH = {MACH_int[i]}, Cy ={val*57.296}") for i, val in enumerate(Cy_int)]
