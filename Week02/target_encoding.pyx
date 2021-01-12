# coding = 'utf-8'
import numpy as np
import pandas as pd
import time
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def target_mean_v1(data, y_name:str, x_name:str):
    cdef:
        cnp.ndarray[cnp.float64_t] result = np.zeros(data.shape[0])

    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def target_mean_v2(data, y_name: str, x_name: str):
    cdef:
        cnp.ndarray[cnp.float64_t] result = np.zeros(data.shape[0])
        dict value_dict = {}
        dict count_dict = {}

    x_val_array = data[x_name].values
    y_val_array = data[y_name].values
    for i in range(data.shape[0]):
        x_val = x_val_array[i]
        y_val = y_val_array[i]
        if x_val_array[i] not in value_dict.keys():
            value_dict[x_val] = y_val
            count_dict[x_val] = 1
        else:
            value_dict[x_val] += y_val
            count_dict[x_val] += 1
    for i in range(data.shape[0]):
        x_val = x_val_array[i]
        y_val = y_val_array[i]
        result[i] = (value_dict[x_val] - y_val) / (count_dict[x_val] - 1)

    return result

def main():
    y = np.random.randint(2, size=(500, 1))
    x = np.random.randint(10, size=(500, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])

    time_start = time.perf_counter()
    result_1 = target_mean_v1(data, 'y', 'x')
    print("Time costs target_mean_v1: " + str(time.perf_counter() - time_start) + "s")
    result_2 = target_mean_v2(data, 'y', 'x')
    print("Time costs target_mean_v2: " + str(time.perf_counter() - time_start) + "s")

    diff = np.linalg.norm(result_1 - result_2)
    print(diff)

if __name__ == '__main__':
    time_start = time.perf_counter()
    main()
    time_end = time.perf_counter()
    print("Time costs total: " + str(time_end - time_start) + "s")
