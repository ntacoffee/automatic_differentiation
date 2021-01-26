#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

x1 = 1.0
x1_chain = {"type": "param", "grad": 0}
x2 = np.pi / 4
x2_chain = {"type": "param", "grad": 0}

y1 = 2 * x1
y1_chain = {"type": "func", "upstream": [x1_chain], "grad": [2]}
y2 = x1 * np.sin(x2)
y2_chain = {
    "type": "func",
    "upstream": [x1_chain, x2_chain],
    "grad": [np.sin(x2), x1 * np.cos(x2)],
}

z = y1 + y2
z_chain = {"type": "func", "upstream": [y1_chain, y2_chain], "grad": [1, 1]}


def backward(chain, grad):
    if chain["type"] == "param":
        chain["grad"] += grad
    elif chain["type"] == "func":
        for i, upstream_chain in enumerate(chain["upstream"]):
            backward(upstream_chain, grad * chain["grad"][i])
    else:
        # 本来はerrorを吐く
        pass


backward(z_chain, 1)

# 可視化
print(x1_chain["grad"])
print(x2_chain["grad"])
