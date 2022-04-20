from rans import encode, decode
import random
import math
import time 

n = 50000000

mean = [random.randint(-256, 256) / 256 for i in range(n)]
scale = [math.exp(10 * random.random() - 5) / 256 for i in range(n)]
msg = [round((mean[i] + scale[i] * (10 * random.random() - 5)) * 256) / 256 for i in range(n)]
# print([m * 256 for m in mean], [m * 256 for m in msg])
x = (1 << 32)

print('start')
t1 = time.time()
x, buf = encode(x, n, msg, mean, scale)
t2 =time.time()
print(x, len(buf))
mean_ = mean[::-1]
scale_ = scale[::-1]
t3 = time.time()
x, msg_rec = decode(x, buf[::-1], n, mean_, scale_)
t4 = time.time()
msg_rec = msg_rec[::-1]

print(x, 1 << 32)
print(f'encode: {t2-t1}', f'decode: {t4-t3}')
# print('decode: ', x, msg_rec)
# print('original: ', 1 << 32, msg)

count = 0
for i in range(n):
    if (msg[i] - msg_rec[i]) > 1e-6:
        print(i, msg[i], msg_rec[i])
        count += 1
print(count)

