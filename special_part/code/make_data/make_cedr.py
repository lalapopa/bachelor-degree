import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy import interpolate
import re
import os 

def sort_by_int_in_str(name):
    try:
        return get_number_from_name(name, '(?<=\=)[\d]*')
    except IndexError:
        return 0 

def get_number_from_name(names, regex):
    try:
        x = iter(names)
    except TypeError:
        try:
            return float(try_to_take_regex(regex, names)) 
        except ValueError:
            return False

    if type(names) == str:
        try:
            return float(try_to_take_regex(regex, names))
        except ValueError:
            return False

    numbers = []
    for name in names: 
        value = try_to_take_regex(regex, name)
        if value:
            value = float(value)
        else:
            continue

        if value in numbers:
            continue 
        else:
            numbers.append(value)
    return sorted(numbers)


def try_to_take_regex(regex, value):
    try:
        return re.findall(regex, value)[0]
    except IndexError:
        return False

def custom_interpolate_Ce_P(M, H, Ce_dr, R_dr, M_int, H_int):
    left_bound, right_bound = get_value_index_where_bounded(H, H_int)
    Ce_target_r = Ce_dr[right_bound]
    Ce_target_l = Ce_dr[left_bound]
    R_dr_r = R_dr[right_bound]
    R_dr_l = R_dr[left_bound]

    Ce_l_int_M = inter_2d_custom(Ce_target_l, M[left_bound], M_int)
    Ce_r_int_M = inter_2d_custom(Ce_target_r, M[right_bound], M_int)
    if (Ce_l_int_M==Ce_l_int_M).all():
        Ce_int = Ce_l_int_M
    else:
        Ce_int = inter_2d_custom(
                np.array([Ce_l_int_M, Ce_r_int_M]),
                np.array([H[left_bound], H[right_bound]]),
                H_int,
                )

    R_l_int_M = inter_2d_custom(R_dr_l, M[left_bound], M_int)
    R_r_int_M = inter_2d_custom(R_dr_r, M[right_bound], M_int)
    if (R_l_int_M==R_r_int_M).all():
        R_int = R_l_int_M
    else:
        R_int = inter_2d_custom(
                np.array([R_l_int_M, R_r_int_M]),
                np.array([H[left_bound], H[right_bound]]),
                H_int,
                )
    return Ce_int, R_int

def inter_2d_custom(array_x, array_y, y_int):
    index_l, index_r = get_value_index_where_bounded(array_y, y_int)
    left_array_x = array_x[index_l]
    right_array_x = array_x[index_r]
    int_values = []
    if index_l == index_r:
        return np.array(array_x[index_l])

    for i in range(0, len(left_array_x)):
        f_ar_x = interpolate.interp1d(
                [array_y[index_l], array_y[index_r]], 
                [left_array_x[i], right_array_x[i]], 
                fill_value="extrapolate"
                )
        int_values.append(f_ar_x(y_int))
    return np.array(int_values)


def get_value_index_where_bounded(array, value):
    array_index_gr= np.unique(np.array(np.where(array >= value)))
    array_index_le = np.unique(np.array(np.where(array <= value)))

    if array_index_gr.size != 0 and array_index_le.size != 0:
        right_bound = array_index_gr[0]
        left_bound = array_index_le[-1]
    else:
        if array_index_le.size == 0:
            right_bound = 0
            left_bound = 0
        else:
            right_bound = len(array)-1
            left_bound = len(array)-1

    return left_bound, right_bound

def get_Ce_dr_P_dr(mach, height)
    folder_path = './raw_data/'
    file_names = os.listdir(folder_path)
    file_names.sort(key=sort_by_int_in_str)

    H_values = np.array(get_number_from_name(file_names, '(?<=\=)[\d]*'))
    Ce_dr_array = []
    R_dr_array = []
    M_array = []
    for file_name in file_names:
        df = pd.read_csv(folder_path+file_name)
        df = df.drop(0)
        columns_name = df.columns.values.tolist()
        M_values = get_number_from_name(columns_name, '(?<=M\=)[\d\.]+')
        M_array.append(M_values)

        R_dr_pos = []
        for M in M_values:
            for i, name in enumerate(columns_name):
                if str(M) in name:
                    R_dr_pos.append(i)
        Ce_dr_pos = [i+1 for i in R_dr_pos]
        Ce_dr_array.append(df.iloc[:, Ce_dr_pos].T.to_numpy().astype(np.float64))
        R_dr_array.append(df.iloc[:, R_dr_pos].T.to_numpy().astype(np.float64))
    M_array = np.array(M_array)
    Ce_dr_array = np.array(Ce_dr_array)
    R_dr_array = np.array(R_dr_array)
    return custom_interpolate_Ce_P(M_array, H_values, Ce_dr_array, R_dr_array, mach, height)

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#
#H = np.arange(0,11,0.05)
#
#Ce_dr = []
#P_dr = []
#for i in H:
#    Ce_dr_H, P_dr_H = custom_interpolate_Ce_P(M_array, H_values, Ce_dr_array, R_dr_array, 0.5, i)
#    Ce_dr.append(Ce_dr_H)
#    P_dr.append(P_dr_H)
#Ce_dr= np.array(Ce_dr)
#P_dr= np.array(P_dr)
#
#H_new = []
#for i, arr in enumerate(Ce_dr):
#    H_row = []
#    for value in arr:
#        H_row.append(H[i])
#    H_new.append(H_row)
#
#H_new = np.array(H_new)
#print(H_new)
#print(Ce_dr)
#print(P_dr)
#
#
#surf = ax.plot_surface(P_dr, Ce_dr, H_new, cmap=cm.coolwarm,
#                               linewidth=0, antialiased=False)
#
#
#ax.set_xlabel('P_dr')
#ax.set_ylabel('Ce_dr')
#ax.set_zlabel('H')
#
#
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()

