import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd

def random_split(values, type):
    min_val = values.min()
    max_val = values.max()
    if type == 0:
        split = np.random.uniform(min_val, max_val)
    else:
        split = np.random.choice(values)
    return split

def performSplit(dataset, attributesIndices, attributes, coltypes, nodeIndices):
    randIndex = np.random.choice(attributesIndices)
    attributesIndices = attributesIndices[attributesIndices != randIndex]
    attribute = attributes[randIndex]
    type = coltypes[attribute]
    data = dataset[nodeIndices, attribute]
    if np.all(data == data[0]):
        return 1, None, None, attributesIndices
    value = random_split(data, type)
    if type == 0:
        left_mask = data < value
    else:
        left_mask = data == value
    left = nodeIndices[left_mask]
    right = nodeIndices[~left_mask]
    return 0, left, right, attributesIndices

def build_tree_and_update(data, nmin, coltypes, nrows, ncols, max_node_size):
    matrix = np.zeros((nrows, nrows))
    nodes = []
    attributes = np.arange(ncols)
    attributes_indices = np.arange(ncols)
    nodeIndices = np.arange(nrows)
    nodes.append({'indices': nodeIndices, 'attributes_indices': attributes_indices})
    while nodes:
        currentNode = nodes.pop()
        nodeIndices = currentNode['indices']
        attributes_indices = currentNode['attributes_indices']
        if len(nodeIndices) <= nmin or len(attributes_indices) == 0:
            if len(nodeIndices) <= max_node_size:
                matrix[np.ix_(nodeIndices, nodeIndices)] += 1.0
            continue
        status, left_indices, right_indices, attributes_indices_new = performSplit(
            data, attributes_indices, attributes, coltypes, nodeIndices)
        if status == 1:
            if len(nodeIndices) <= max_node_size:
                matrix[np.ix_(nodeIndices, nodeIndices)] += 1.0
        else:
            if len(left_indices) > 0:
                nodes.append({'indices': left_indices, 'attributes_indices': attributes_indices_new.copy()})
            if len(right_indices) > 0:
                nodes.append({'indices': right_indices, 'attributes_indices': attributes_indices_new.copy()})
    return matrix

def build_trees_batch(data, nmin, coltypes, nrows, ncols, count, max_node_size):
    matrix = np.zeros((nrows, nrows))
    for _ in range(count):
        matrix += build_tree_and_update(data, nmin, coltypes, nrows, ncols, max_node_size)
    return matrix

def getSim(data, nmin, coltypes, nTrees, max_node_size):
    nrows, ncols = data.shape
    num_cores = multiprocessing.cpu_count()
    trees_per_core = nTrees // num_cores
    remaining_trees = nTrees % num_cores
    tree_counts = [trees_per_core] * num_cores
    for i in range(remaining_trees):
        tree_counts[i] += 1
    matrices = Parallel(n_jobs=num_cores)(
        delayed(build_trees_batch)(data, nmin, coltypes, nrows, ncols, count, max_node_size)
        for count in tree_counts
    )
    matrix = sum(matrices) / nTrees
    return matrix

def readCSV(filename, sep):
    try:
        # Read the file using pandas and return a numpy array
        data = pd.read_csv(filename, sep=sep, header=None).to_numpy()
        return data
    except Exception as e:
        raise ValueError(f"Error reading the file: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='An implementation of UET')
    parser.add_argument('-p', '--path', type=str, required=True, help='Data path')
    parser.add_argument('-s', '--sep', type=str, default='\t', help='Separator')
    parser.add_argument('-c', '--ctypes', type=str, default='0,', help='Coltypes')
    parser.add_argument('-n', '--nmin', type=float, default=0.33, help='Nmin')
    parser.add_argument('-t', '--ntrees', type=int, default=500, help='Number of trees')
    parser.add_argument('-m', '--massbased', type=int, default=0, help='Mass-based dissimilarity')
    parser.add_argument('-o', '--optimize', type=int, default=0, help='Find optimal parameters')
    parser.add_argument('--max_node_size', type=int, default=1000, help='Maximum node size to update similarity matrix')
    args = parser.parse_args()
    
    path = args.path
    sep = args.sep
    coltypesString = args.ctypes
    nminPercent = args.nmin
    nTrees = args.ntrees
    massBased = args.massbased
    max_node_size = args.max_node_size
    
    data = readCSV(path, sep)
    nrows, ncols = data.shape
    nmin = int(np.floor(nminPercent * nrows))
    if coltypesString.endswith(','):
        coltype = int(coltypesString[0])
        coltypes = [coltype] * ncols
    else:
        coltypes = [int(c) for c in coltypesString.split(',') if c != '']
    coltypes = np.array(coltypes)
    
    from time import time
    startTime = time()
    if massBased == 0:
        matrix = getSim(data, nmin, coltypes, nTrees, max_node_size)
    else:
        pass
    endTime = time()
    print("Time: {:.2f}ms".format((endTime - startTime) * 1000))
    np.savetxt("matrix_uet.csv", matrix, delimiter='\t')
    
if __name__ == '__main__':
    main()
