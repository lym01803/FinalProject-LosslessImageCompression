# distutils: language=c++

import cython
from libcpp.vector cimport vector

cdef extern from 'cmath':
    float log(float)
cdef extern from 'cmath':
    float exp(float)
cdef extern from 'cmath':
    float round(float)

cdef unsigned long long L = 0x100000000
cdef unsigned long long logL = 32
cdef unsigned long long Mask = 0xffffffff
cdef unsigned long long B = 0x100000000
cdef unsigned long long logB = 32
cdef unsigned long long M = 0x1000000
cdef unsigned long long logM = 24
cdef int nbits = 8
cdef int nbits_bins = 256
cdef int nbins = 2048
# (L, BL-1)

cdef logistic(x: cython.float):
    return 1.0 / (1.0 + exp(-x))

cdef inv_logistic(x: cython.float):
    return log(x / (1.0 - x))

cdef CDF(x: cython.float, mean: cython.float, scale: cython.float, lower: cython.float):
    cdef int part1, part2
    part2 = int(round((x - lower) * 256))
    part1 = int(round(logistic((x + 0.5 / 256 - mean) / scale) * (M - 2048)))
    return part1 + part2

def encode(state: cython.ulonglong, n: cython.int, list x_, list mean_, list scale_):
    cdef vector[float] x = x_
    cdef vector[float] mean = mean_ 
    cdef vector[float] scale = scale_
    cdef vector[unsigned long long] cdf
    cdef vector[unsigned long long] freq
    cdef vector[unsigned int] buffer
    cdef float lower
    cdef int start
    cdef int end
    cdef int i

    i = 0
    while i < n:
        lower = round(mean[i] * 256 - 1024) / 256
        start = CDF(x[i] - 1. / 256, mean[i], scale[i], lower)
        end = CDF(x[i], mean[i], scale[i] ,lower)
        cdf.push_back(start)
        freq.push_back(end - start)
        i += 1
    
    i = 0
    # Range [ls, ls * B), where ls = freq / M * L
    # ls * B = freq * L / M * B = freq << (32 - 24 + 32) = freq << 40
    while i < n:
        if state >= (freq[i] << 40):
            buffer.push_back(state & Mask)
            state >>= 32
        state = ((state // freq[i]) << 24) + (state % freq[i]) + cdf[i] 
        i += 1
    return state, buffer

def decode(state: cython.ulonglong, list buffer_, n: cython.int, list mean_, list scale_):
    cdef vector[float] mean = mean_
    cdef vector[float] scale = scale_
    cdef vector[unsigned int] buffer = buffer_
    cdef vector[float] message
    cdef int i 
    cdef int lower 
    cdef float lower_f
    cdef int upper
    cdef int s
    cdef int pos
    cdef unsigned long long cdf_s
    cdef unsigned long long freq_s
    cdef unsigned long long mod
    pos = 0
    i = 0
    # The buffer should to be reversed
    while i < n:
        if state < L:
            state = ((state << 32) | buffer[pos])
            pos += 1
        mod = (state & 0xffffff)
        # calculate s
        lower = int(round(mean[i] * 256 - 1024))
        upper = lower + 2047
        lower_f = lower / 256.
        # print(mod, lower, upper, lower_f)
        while lower <= upper:
            s = (lower + upper) >> 1
            cdf_s = CDF(s / 256., mean[i], scale[i], lower_f)
            # print(mod, lower, upper, s, cdf_s)
            if cdf_s > mod:
                upper = s - 1
            else:
                lower = s + 1
        s = lower
        message.push_back(s / 256.)
        cdf_s = CDF((s - 1) / 256., mean[i], scale[i], lower_f)
        freq_s = CDF(s / 256., mean[i], scale[i], lower_f) - cdf_s
        state = (state >> 24) * freq_s + (state & 0xffffff) - cdf_s
        i += 1
    return state, message

