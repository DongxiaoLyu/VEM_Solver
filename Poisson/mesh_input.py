import numpy as np

"""
mesh input
form:
    Nodes
    <id> <x> <y>
    ...

    Elements
    <id> <node_id_1> <node_id_2> ... <node_id_n>
"""

def read_mesh(filename):
    nodes = []
    elems = []

    section = None # to be "nodes" or "element"

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # indentify section title
            lower = line.lower()
            if lower == 'nodes':
                section = 'nodes'
                continue
            elif lower == 'elements':
                section = 'elements'
                continue

            parts = line.split()
            if section == 'nodes':
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                nodes.append([x, y])

            elif section == 'elements':
                eid = int(parts[0])
                ng = [int(p) - 1 for p in parts[1:]] # nodes group
                elems.append(np.array(ng))

            else:
                raise ValueError(f"Undefined section, please check the format in input mesh file")

    return nodes, elems

if __name__ == "__main__":
    nodes, elems = read_mesh('voronoi_mesh.txt')
    print("Nodes:")
    for i, node in enumerate(nodes):
        print(f"Node {i}: {node}")
    print("\nElements:")
    for i, elem in enumerate(elems):
        print(f"Element {i}: {elem}")


