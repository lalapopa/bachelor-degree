import numpy as np

import config
from lerp import linear1d
from DataHandler import DataHandler as dh

class AerodynamicsData:
    def __init__(self):
        self.df = dh(config.PATH_DATA+"ad_data.csv")

        self.mach_A_column = self.df.get_column("M")
        self.A_column = self.df.get_column("A")

        self.mach_Cxm_column = self.df.get_column("M")
        self.Cxm_column = self.df.get_column("Cxm")

        self.mach_Cym_column = self.df.get_column("M")
        self.Cym_column = self.df.get_column("Cym")

        self.Cedr_column = self.df.get_column("Cedr")
        self.Rdr_column = self.df.get_column("Rdr")

        column_x_text = "Ptilda"
        column_mach_text = "M_H_ptilda"
        self.Ptilda_dict = self._take_from_table_in_dict(
            column_mach_text, column_x_text
        )

        column_x_text = "Cetilda"
        column_mach_text = "M_H_ce"
        self.Cetilda_dict = self._take_from_table_in_dict(
            column_mach_text, column_x_text
        )

    def PTilda(self, mach, height):
        index = self._take_range(height)
        values = []
        for i in index:
            text = "Ptilda" + str(i)
            values.append(
                linear1d(
                    self.Ptilda_dict.get("M_H_ptilda"), self.Ptilda_dict.get(text), mach
                )
            )
        H_s = [1000 * val for val in index]
        result = linear1d(H_s, values, height)
        return result

    def CeTilda(self, mach, height):
        index = self._take_range(height)
        values = []
        for i in index:
            text = "Cetilda" + str(i)
            values.append(
                linear1d(
                    self.Cetilda_dict.get("M_H_ce"), self.Cetilda_dict.get(text), mach
                )
            )
        H_s = [1000 * val for val in index]
        result = linear1d(H_s, values, height)
        return result

    def A_value(self, mach):
        return linear1d(self.mach_A_column, self.A_column, mach)

    def Cxm_value(self, mach):
        return linear1d(self.mach_Cxm_column, self.Cxm_column, mach)

    def Cym_value(self, mach):
        return linear1d(self.mach_Cym_column, self.Cym_column, mach)

    def Ce_dr_value(self, R):
        return linear1d(self.Rdr_column, self.Cedr_column, R)

    def _take_from_table_in_dict(self, x_name, y_name):
        output = {}
        x_column = self.df.get_column(x_name)
        H_text = (0, 2, 4, 6, 8, 10, 11)
        output[x_name] = x_column
        for h in H_text:
            y_text = y_name + str(h)
            y_column = self.df.get_column(y_text)
            output[y_text] = y_column
        return output

    def _take_range(self, H):
        H_text = [0, 2, 4, 6, 8, 10, 11]
        H_val = [val * 1000 for val in H_text]
        if H >= H_val[-1]:
            return (H_text[-2], H_text[-1])
        if H <= H_val[0]:
            return (H_text[0], H_text[1])
        for i, val in enumerate(H_val):
            if val > H:
                return (H_text[i - 1], H_text[i])
