import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib
import numpy as np


def pgf_setting():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
                        {
    "pgf.texsystem": "pdflatex",
    "font.family": "sans-serif",
    "text.usetex": True,
    "font.size": 12,
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join(
        [
            r"\usepackage[warn]{mathtext}",
            r"\usepackage[T2A]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[english,russian]{babel}",
        ]
        ),
        }
    )

pgf_setting()
H_const = 627.6 
H_300 = 40.93
flight_numbers = 100
path = '/home/lalapopa/Documents/uni/4_course/2_sem/diploma_work/presentation/figures/'

x_axis = np.arange(1,100+1)
fuel_lost_H_const = [H_const*i for i in x_axis]
fuel_lost_H_300 = [H_300*i for i in x_axis]

fig, ax = plt.subplots()

plt.plot(x_axis, fuel_lost_H_const, linewidth=2, label='Полет на постоянной высоте')
plt.plot(x_axis, fuel_lost_H_300, linewidth=2, label='Эшелонированный полет')
ax.legend(ncol=4, loc=3, bbox_to_anchor=(0, 1), frameon=False)
ax.xaxis.set_minor_locator(matplotlib.ticker.MaxNLocator(25))
ax.grid(color="gray", which="major", axis="x",linestyle='--', linewidth=0.5)
ax.grid(color="gray", which="major", axis="y",linestyle='--', linewidth=0.5)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.yscale('log', base=10)
plt.xlabel('Количество полетов')
plt.ylabel('Избыток топлива, [кг]')
plt.xlim([0, x_axis[-1]])
plt.ylim([1, 10**5])
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())
plt.savefig(path+'funny_plot.pgf')


