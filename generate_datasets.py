import numpy as np
import os

DIRECTORY = os.path.realpath(os.path.dirname(__file__))

SEED = 4136121025
DISTRIBUTIONS = [
    ("exp", 0.3),
    ("exp", 0.4),
    ("exp", 0.6),
    ("exp", 0.8),
    ("exp", 0.9),
    ("exp", 1.15),
    ("exp", 1.5),
    ("exp", 1.8),
    ("wei", 0.8, 0.4),
    ("wei", 0.9, 0.5),
    ("wei", 0.9, 0.7),
    ("wei", 0.9, 1.1),
    ("wei", 0.9, 1.5),
    ("wei", 1.0, 1.1),
    ("wei", 1.0, 1.9),
    ("wei", 1.3, 0.5),
    ("lgn", 0.1, 1.0),
    ("lgn", 0.2, 0.75),
    ("lgn", 0.3, 0.3),
    ("lgn", 0.3, 0.5),
    ("lgn", 0.3, 0.8),
    ("lgn", 0.4, 0.32),
    ("lgn", 0.5, 0.3),
    ("lgn", 0.5, 0.7),
    ("gam", 0.2, 0.75),
    ("gam", 0.3, 1.3),
    ("gam", 0.3, 2.0),
    ("gam", 0.5, 1.5),
    ("gam", 0.8, 1.0),
    ("gam", 0.9, 1.3),
    ("gam", 1.4, 0.9),
    ("gam", 1.5, 0.7),
]

np.random.seed(SEED)

def generate_tree(d, features):
    if d == 0:
        return DISTRIBUTIONS[np.random.randint(len(DISTRIBUTIONS))]

    f = [*features][np.random.randint(len(features))]
    features.remove(f)

    left_child = generate_tree(d - 1, features)
    right_child = generate_tree(d - 1, features)

    features.add(f)

    return (f, left_child, right_child)

def traverse_tree(tree, instance):
    if type(tree[0]) == str:
        if tree[0] == "exp":
            return np.random.exponential(tree[1])
        if tree[0] == "wei":
            return np.random.weibull(tree[1]) * tree[2]
        if tree[0] == "lgn":
            return np.random.lognormal(tree[1], tree[2])
        if tree[0] == "gam":
            return np.random.gamma(tree[1], tree[2])

    f = tree[0]
    value = instance[2 + f]
    child = tree[1 + value]
    return traverse_tree(child, instance)

def generate_dataset(n, f, c):
    instances = [[0, 0] + [np.random.randint(0, 2) for _ in range(f)] for _ in range(n)]

    tree = generate_tree(5, {*range(f)})
    ks = [1e9]
    for inst in instances:
        time = traverse_tree(tree, inst)
        u = 1 - np.random.random() ** 2

        inst[0] = time
        inst[1] = u
        ks.append(time / u)

    ks = sorted(ks)
    k = ks[int(n * (1 - c))]

    for inst in instances:
        censor = k * inst[1]

        if censor < inst[0]:
            inst[0] = censor
            inst[1] = 0
        else:
            inst[1] = 1

    return instances

if __name__ == "__main__":
    SETTINGS = [
        (n, f, c, i)
            for n in [100, 500, 1000, 5000, 10000]
            for f in [10, 50, 100]
            for c in [10, 50, 80]
            for i in range(5)
    ]

    for n, f, c, i in SETTINGS:
        print(f"\033[35;1mGenerating dataset {n}_{f}_{c}_{i} ...\033[0m")

        instances = generate_dataset(n, f, c / 100)

        file = open(f"{DIRECTORY}\\datasetsSA\\dataset_{n}_{f}_{c}_{i}.txt", "w")
        for inst in instances:
            file.write(" ".join(str(j) for j in inst))
            file.write("\n")
        file.close()
