#!/bin/python3

import matplotlib.pyplot as plt
import sys


file = open(sys.argv[1], "r")

vals = []
for line in file:
    val = float(line[0:len(line) - 1])
    vals.append(val)

plt.plot(vals)
plt.show()
