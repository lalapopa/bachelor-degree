import pandas as pd
import json
import numpy as np

from DataHandler import DataHandler as dh
from plane_data import PlaneData
from main import optimal_fly_param, to_height_mach_table


def create_json_with_opt_fly_params(file_name):
    H_array = np.arange(7000, 12000, 500)
    mass_array = np.array(
        [125000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000]
    )

    result_table = {}
    for mass in mass_array:
        mass_table = {}

        H_fly, V_opt, M_opt, q_km_min = optimal_fly_param(
            mass, H_array, like_array=True
        )

        min_fuel_index = dh.get_min_or_max(q_km_min)

        mass_table = to_height_mach_table(H_fly, M_opt, q_km_min, V_opt, min_fuel_index)
        result_table[str(mass)] = mass_table

        print(f'{"#"*10} H={H_fly[min_fuel_index]} feet, m={mass} kg {"#"*10}')
        print("_" * 30)
        print(f"Consumption {q_km_min[min_fuel_index]}")
        print(f"MACH = {M_opt[min_fuel_index]}")
        print("_" * 30)

    with open(file_name, "w", encoding="utf-8") as jf:
        json.dump(result_table, jf, indent=4)


def parse_json_and_return_latex_table(file_name):
    with open(file_name) as jf:
        data = json.load(jf)

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


json_name = "output.json"
output_latex_name = "table.tex"

create_json_with_opt_fly_params(json_name)

with open(output_latex_name, "w") as f:
    f.write(parse_json_and_return_latex_table(json_name))
