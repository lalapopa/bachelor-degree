import pandas as pd
import json
import numpy as np
import os
import re

from DataHandler import DataHandler as dh
from plane_data import PlaneData
from main import optimal_fly_param
import config


def to_height_mach_table(H, M, q_km, V, index):
    table = []
    for i, val in enumerate(H):
        if index == i:
            add_text = "\\cellcolor{green}"
        else:
            add_text = ""
        only_H = {}
        only_H["$M$"] = str(f"{M[i]}{add_text}")
        only_H["$q_{km}$"] = str(f"{q_km[i]}{add_text}")
        only_H["$V$"] = str(f"{V[i]}{add_text}")
        only_H["$H$"] = str(val)
        table.append(only_H)
    return table


def create_json_with_opt_fly_params(file_name):
    H_array = np.arange(7000, 13000, 500)
    #    H_array = np.arange(7000, 13000, 100)
    mass_array = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190]) * 1000
    # mass_array = np.array([140, 142, 148, 150]) * 1000

    result_table = {}
    for mass in mass_array:
        mass_table = {}

        H_fly, V_opt, M_opt, q_km_min = optimal_fly_param(
            mass, H_array, like_array=True
        )

        min_fuel_index = dh.get_min_or_max(q_km_min)

        q_km_min = np.array(["-" if i == 999 else str(round(i, 3)) for i in q_km_min])
        H_fly = np.array(["-" if i == 999 else str(round(i, 0)) for i in H_fly])
        M_opt = np.array(["-" if i == 999 else str(round(i, 3)) for i in M_opt])
        V_opt = np.array(["-" if i == 999 else str(round(i, 3)) for i in V_opt])

        mass_table = to_height_mach_table(H_fly, M_opt, q_km_min, V_opt, min_fuel_index)
        result_table[str(mass)] = mass_table
        print(f'{"#"*10} H={H_fly[min_fuel_index]} feet, m={mass} kg {"#"*10}')
        print("_" * 30)
        print(f"Consumption {q_km_min[min_fuel_index]}")
        print(f"MACH = {M_opt[min_fuel_index]}")
        print(f"V = {V_opt[min_fuel_index]}")
        print("_" * 30)

    with open(file_name, "w", encoding="utf-8") as jf:
        json.dump(result_table, jf, indent=4)


def parse_json_and_return_latex_table(file_name):
    with open(file_name) as jf:
        data = json.load(jf)
    os.remove(file_name)
    keys_data = list(data.keys())[0]
    actual_name = [d["$H$"] for d in data[keys_data]]
    mass_array = []
    frame = []
    for mass, d in data.items():
        mass_array.append(str(int(mass) / 1000))
        clear_dict = []
        for raw_dict in d:
            d1 = raw_dict
            del d1["$H$"]
            clear_dict.append(d1)
        frame.append(pd.DataFrame(d).T)
    df = pd.concat(frame, keys=mass_array)
    col_name = [name for name in df.columns]
    new_name = dict(zip(col_name, actual_name))
    df = df.rename(columns=new_name)
    return df.to_latex(float_format="%.3f", escape=False)


def format_latex_table(latex_table):
    latex_table = re.sub(
        r"(\\bottomrule)|(\\midrule)|(\\toprule)", r"\\hline", latex_table
    )
    number_of_column = len(list(re.findall(r"(?<={)[lr]+(?=})", latex_table))[0])
    table_argument = "|"
    for i, num in enumerate(range(0, number_of_column)):
        if i < 2 or i == number_of_column - 1:
            table_argument += "l|"
        else:
            table_argument += "l"
    latex_table = re.sub(r"(?<={)[lr]+(?=})", table_argument, latex_table)

    main_table = latex_table.splitlines()[3:]
    header_with_H = latex_table.splitlines()[2]
    header = (
        "\multicolumn{2}{|c|}{$m$, тонн}& \multicolumn{%s}{c|}{$H$, м}\\\ \n \cline{3-%s}\n"
        % (str(number_of_column - 2), str(number_of_column))
    )
    header_with_H = re.sub(r"([ ]+&)(\1)", r"\\multicolumn{2}{|c|}{}&", header_with_H)

    counter = 0
    match_H = r"^\d+"
    formated_latex_table = (
        "".join(latex_table.splitlines()[0:2]) + "\n" + header + header_with_H
    )
    for i, line in enumerate(main_table):
        if re.search(match_H, line) or i > len(main_table) - 4:
            counter = 0
        else:
            counter += 1
        if counter == 2:
            new_line = "\n" + line + "\n\hline"
        else:
            new_line = "\n" + line
        formated_latex_table += f"{new_line}"
    return formated_latex_table


json_name = "output.json"
output_latex_name = "table.tex"
create_json_with_opt_fly_params(json_name)

with open(config.PATH_TABLE + output_latex_name, "w") as f:
    latex_table = parse_json_and_return_latex_table(json_name)
    latex_table_formated = format_latex_table(latex_table)
    f.write(latex_table_formated)
