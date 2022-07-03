import numpy as np

if __name__ == '__main__':
    cfg = {"name": "rectangle", "shape": {"Flange length": 1, "Web length": 3}}
    # Rc = mb.Rectangle(cfg)
    # print(Rc.yz)
a = {'name': 1, "type": {'square': 1}}
print(a["name"])
a = np.array([1, 2, 3])
print(np.concatenate([a for i in range(3)], axis = 0))
print(np.hstack([a, a]))