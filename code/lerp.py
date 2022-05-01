import numpy as np


def linear1d(x, y, x_int):
    if isinstance(x, np.ndarray):
        pass
    else:
        x = np.array(x)

    if isinstance(y, np.ndarray):
        pass
    else:
        y = np.array(y)

    y_int = []
    if hasattr(x_int, "__iter__"):
        for i, x_val in enumerate(x_int):
            if x_val in x:
                index = np.where(x == x_val)
                y_int.append(y[int(index[0])])
                continue
            else:
                x_pair, y_pair = find_pair(x, y, x_val)
                y_int.append(lerp(x_pair, y_pair, x_val))
    else:
        if x_int in x:
            index = np.where(x == x_int)
            y_int.append(y[int(index[0])])
        else:
            x_pair, y_pair = find_pair(x, y, x_int)
            y_int.append(lerp(x_pair, y_pair, x_int))

    if len(y_int) == 1:
        return y_int[0]
    else:
        return y_int


def find_pair(x_array, y_array, element):
    if element < x_array[0]:
        pair_x = [x_array[0], x_array[1]]
        pair_y = [y_array[0], y_array[1]]
    elif element > x_array[-1]:
        pair_x = [x_array[-2], x_array[-1]]
        pair_y = [y_array[-2], y_array[-1]]

    else:
        for i, x in enumerate(x_array):
            if element < x and i != 0:
                pair_x = [x_array[i - 1], x]
                pair_y = [y_array[i - 1], y_array[i]]
                break
    return pair_x, pair_y


def lerp(x, y, x_int):
    return y[0] + (x_int - x[0]) * ((y[1] - y[0]) / (x[1] - x[0]))
