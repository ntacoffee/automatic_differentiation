#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

x1 = 1.0
x1_chain = {"requires_grad": True, "grad": 0, "backward_branches": []}
x2 = np.pi / 4
x2_chain = {"requires_grad": True, "grad": 0, "backward_branches": []}

y1 = 2 * x1
y1_chain = {
    "requires_grad": True,
    "grad": 0,
    "backward_branches": [{"chain": x1_chain, "f_grad": 2}],
}
y2 = x1 * np.sin(x2)
y2_chain = {
    "requires_grad": True,
    "grad": 0,
    "backward_branches": [
        {"chain": x1_chain, "f_grad": np.sin(x2)},
        {"chain": x2_chain, "f_grad": x1 * np.cos(x2)},
    ],
}

z = y1 + y2
z_chain = {
    "requires_grad": True,
    "grad": 0,
    "backward_branches": [
        {"chain": y1_chain, "f_grad": 1},
        {"chain": y2_chain, "f_grad": 1},
    ],
}


def backward(chain, grad=1):

    if chain["requires_grad"]:
        chain["grad"] += grad

    if not chain["backward_branches"]:
        # reached the end of the chain
        return

    for branch in chain["backward_branches"]:
        backward(branch["chain"], grad * branch["f_grad"])


backward(z_chain)

# 可視化
print(x1_chain["grad"])
print(x2_chain["grad"])
