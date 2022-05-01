import cProfile
import pstats
import time

from my_integral_calc import trapezoid
from lerp import linear1d
from plane_data import PlaneData
from aero_data import AeroData
from DataHandler import DataHandler as dh
from main import optimal_fly_param

plane_data_file_name = "var_data.csv"
il_76 = PlaneData(plane_data_file_name, 12)

m0 = il_76.MTOW
m_k = m0 - il_76.TFL

start = time.time()
with cProfile.Profile() as pr:
    result = optimal_fly_param(m0)
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()

print(f"DONE! Run time = {time.time() - start}")
print(f"function result is {result}")
