from DataHandler import DataHandler as dh
import numpy as np

class PlaneData:
    def __init__(self):
        df = dh("var_data.csv")
        self.M_OGR = float(df.get_column("M_OGR"))
        self.V_I_MAX = float(df.get_column("V_i"))
        self.OEW = 88500  # operating empty weight in KG
        self.TFL = 60000 # total fuel load in KG 
        self.TP = 20000 # total payload KG
        self.MTOW = 166000 # maximum takeoff weight KG
        self.OTN_M_TSN = float(df.get_column("otn_m_tsn"))
        self.OTN_M_T = float(df.get_column("otn_m_t"))
        self.OTN_M_CH = float(df.get_column("otn_m_CH"))
        self.OTN_P_0 = float(df.get_column("otn_p_0"))
        self.CE_0 = float(df.get_column("Ce0"))
        self.N_DV = float(df.get_column("n_dv"))
        self.N_REV = float(df.get_column("n_rev"))
        self.PS = float(df.get_column("ps"))
        self.B_A = float(df.get_column("ba"))
        self.OTN_L_GO = float(df.get_column("otn_L_go"))
        self.WING_AREA = np.array([300])
