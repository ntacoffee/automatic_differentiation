#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

x1 = 1.0
x1_node = {"requires_grad": True, "grad": 0, "backward_branches": []}
x2 = np.pi / 4
x2_node = {"requires_grad": True, "grad": 0, "backward_branches": []}

y1 = 2 * x1
y1_node = {
    "requires_grad": True,
    "grad": 0,
    "backward_branches": [{"previous_node": x1_node, "f_grad": 2}],
}
y2 = x1 * np.sin(x2)
y2_node = {
    "requires_grad": True,
    "grad": 0,
    "backward_branches": [
        {"previous_node": x1_node, "f_grad": np.sin(x2)},
        {"previous_node": x2_node, "f_grad": x1 * np.cos(x2)},
    ],
}

z = y1 + y2
z_node = {
    "requires_grad": True,
    "grad": 0,
    "backward_branches": [
        {"previous_node": y1_node, "f_grad": 1},
        {"previous_node": y2_node, "f_grad": 1},
    ],
}


def backward(node, grad=1):

    if node["requires_grad"]:
        node["grad"] += grad

    if not node["backward_branches"]:
        # reached the end of the chain
        return

    for backward_branch in node["backward_branches"]:
        backward(backward_branch["previous_node"], grad * backward_branch["f_grad"])


backward(z_node)

# 可視化
print(x1_node["grad"])
print(x2_node["grad"])
