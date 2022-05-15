import cProfile
import pstats
import time
import numpy as np

from lerp import linear1d
from plane_data import PlaneData
from aerodynamics_data import AerodynamicsData
from DataHandler import DataHandler as dh
from main import optimal_fly_param





start = time.time()
with cProfile.Profile() as pr:
    result =
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()

print(f"DONE! Run time = {time.time() - start}")
print(f"function result is {len(result)}")
