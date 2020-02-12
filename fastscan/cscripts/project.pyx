cimport numpy as np
import numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_i



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def project_r0(
        np.ndarray[DTYPE_i, ndim = 1]  spos,
        np.ndarray[DTYPE_t, ndim = 1]  signal,
        np.ndarray[DTYPE_t, ndim = 1]  dark_control,
        np.ndarray[DTYPE_t, ndim = 1]  reference,
        bint use_dark_control):
    cdef int n_pts, pos_min, res_size, i, pos, count
    cdef float r0, val, ref, r0_

    n_pts = len(spos)
    pos_min = spos.min()
    res_size = spos.max() - pos_min + 1

    cdef np.ndarray[DTYPE_t, ndim = 1] result_val = np.zeros(res_size, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] result_ref = np.zeros(res_size, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] result_normref = np.zeros(res_size, dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim = 1] norm_array = np.zeros(res_size, dtype=DTYPE)


    if use_dark_control:
        for i in range(n_pts//2):
            pos = int((spos[2 * i] + spos[2 * i + 1]) // 2) - pos_min
            if dark_control[2 * i] > dark_control[2 * i + 1]:
                val = signal[2 * i] - signal[2 * i + 1]
                ref = reference[2 * i]
            else:
                val = signal[2 * i + 1] - signal[2 * i]
                ref = reference[2 * i + 1]

            result_val[pos] += val
            result_ref[pos] += ref
            norm_array[pos] += 1

    else:

        for i in range(n_pts):
            result_val[spos[i]-pos_min] += signal[i]
            result_ref[spos[i]-pos_min] += reference[i]
            norm_array[spos[i]-pos_min] += 1.
    print(np.sum(result_ref),np.sum(norm_array))

    count = 0
    for i in range(len(norm_array)):
        if norm_array[i] != 0:
            r0 += result_ref[i]/norm_array[i]
            count += 1
    r0 = r0/count
    print(r0)

    # r0 = np.mean(np.nan_to_num(result_ref/norm_array))
    return result_val/(norm_array*r0)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def project(
        np.ndarray[DTYPE_i, ndim = 1]  spos,
        np.ndarray[DTYPE_t, ndim = 1]  signal,
        np.ndarray[DTYPE_t, ndim = 1]  dark_control,
        bint use_dark_control):
    cdef int n_pts, pos_min, res_size, i, pos
    cdef float r0, val,ref

    n_pts = len(spos)
    pos_min = spos.min()
    res_size = spos.max() - pos_min + 1

    cdef np.ndarray[DTYPE_t, ndim = 1] result_val = np.zeros(res_size, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] norm_array = np.zeros(res_size, dtype=DTYPE)


    if use_dark_control:
        for i in range(n_pts//2):
            pos = int((spos[2 * i] + spos[2 * i + 1]) // 2) - pos_min
            if dark_control[2 * i] > dark_control[2 * i + 1]:
                val = signal[2 * i] - signal[2 * i + 1]
            else:
                val = signal[2 * i + 1] - signal[2 * i]

            result_val[pos] += val
            norm_array[pos] += 1

    else:

        for i in range(n_pts):
            result_val[spos[i]-pos_min] += signal[i]
            norm_array[spos[i]-pos_min] += 1.

    return result_val/(norm_array)
