import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

TARGET = "slab_released"


def questions(tree, names):
    """Split features from the root down, in the order the tree asks them."""
    t = tree.tree_
    internal = lambda i: t.children_left[i] != -1

    # usual case: a chain, every question has one leaf side and one side that asks again
    path, node = [], 0
    while internal(node):
        path.append(names[t.feature[node]])
        kids = [c for c in (t.children_left[node], t.children_right[node]) if internal(c)]
        if len(kids) != 1:
            break
        node = kids[0]
    else:
        return path
    if not kids:
        return path

    # branching tree: follow the path to the deepest 'released' leaf
    pos = list(tree.classes_).index(1)
    depth = {0: 0}
    for i in range(t.node_count):
        for c in (t.children_left[i], t.children_right[i]):
            if c != -1:
                depth[c] = depth[i] + 1
    leaves = [i for i in range(t.node_count) if not internal(i)]
    released = [i for i in leaves if np.argmax(t.value[i][0]) == pos]
    target = max(released, key=lambda i: (depth[i], t.n_node_samples[i]))

    parent = {}
    for i in range(t.node_count):
        for c in (t.children_left[i], t.children_right[i]):
            if c != -1:
                parent[c] = i
    path, node = [], target
    while node in parent:
        node = parent[node]
        path.append(names[t.feature[node]])
    return path[::-1]


def main():
    df = pd.read_csv("test-input.csv")
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    tree = DecisionTreeClassifier(random_state=0).fit(X, y)
    word = "".join(f[0] for f in questions(tree, list(X.columns))).upper()
    with open("test-output.txt", "w") as f:
        f.write(f"SIGMOID_{word}\n")


if __name__ == "__main__":
    main()
