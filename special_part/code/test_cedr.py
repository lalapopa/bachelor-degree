from main import Calculation
from plane_data import PlaneData

H = 10900

il_76 = PlaneData()
m0 = il_76.MTOW

calc = Calculation(120000, il_76, H)
q, M, V = calc.find_min_fuel_consumption()




