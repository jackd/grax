def nested_items(tree):
    for k0, v0 in tree.items():
        for k1, v1 in v0.items():
            yield ((k0, k1), v1)
