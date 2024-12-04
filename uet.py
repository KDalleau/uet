import numpy as np
from joblib import Parallel, delayed
import multiprocessing

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

def build_tree_and_update(data, nmin, coltypes, nrows, ncols):
    matrix = np.zeros((nrows, nrows))
    nodes = []
    attributes = np.arange(ncols)
    attributes_indices = np.arange(ncols)
    nodeIndices = np.arange(nrows)
    status, left_indices, right_indices, attributes_indices_new = performSplit(
        data, attributes_indices, attributes, coltypes, nodeIndices)
    if status == 1:
        matrix[nodeIndices[:, None], nodeIndices] += 1.0
        return matrix
    if len(left_indices) < nmin:
        matrix[left_indices[:, None], left_indices] += 1.0
    else:
        nodes.append({'indices': left_indices, 'attributes_indices': attributes_indices_new})
    if len(right_indices) < nmin:
        matrix[right_indices[:, None], right_indices] += 1.0
    else:
        nodes.append({'indices': right_indices, 'attributes_indices': attributes_indices_new})
    while nodes:
        currentNode = nodes.pop()
        if len(currentNode['attributes_indices']) == 0:
            indices = currentNode['indices']
            matrix[indices[:, None], indices] += 1.0
            continue
        nodeIndices = currentNode['indices']
        attributes_indices = currentNode['attributes_indices']
        status, left_indices, right_indices, attributes_indices_new = performSplit(
            data, attributes_indices, attributes, coltypes, nodeIndices)
        if status == 1:
            matrix[nodeIndices[:, None], nodeIndices] += 1.0
        else:
            if len(left_indices) < nmin:
                matrix[left_indices[:, None], left_indices] += 1.0
            else:
                nodes.append({'indices': left_indices, 'attributes_indices': attributes_indices_new})
            if len(right_indices) < nmin:
                matrix[right_indices[:, None], right_indices] += 1.0
            else:
                nodes.append({'indices': right_indices, 'attributes_indices': attributes_indices_new})
    return matrix

def getSim(data, nmin, coltypes, nTrees):
    nrows, ncols = data.shape
    num_cores = multiprocessing.cpu_count()
    trees_per_core = nTrees // num_cores
    remaining_trees = nTrees % num_cores
    tree_counts = [trees_per_core] * num_cores
    for i in range(remaining_trees):
        tree_counts[i] += 1
    matrices = Parallel(n_jobs=num_cores)(
        delayed(build_trees_batch)(data, nmin, coltypes, nrows, ncols, count)
        for count in tree_counts
    )
    matrix = sum(matrices) / nTrees
    return matrix

def build_trees_batch(data, nmin, coltypes, nrows, ncols, count):
    matrix = np.zeros((nrows, nrows))
    for _ in range(count):
        matrix += build_tree_and_update(data, nmin, coltypes, nrows, ncols)
    return matrix

def readCSV(filename, sep):
    data = np.loadtxt(filename, delimiter=sep)
    return data

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
    args = parser.parse_args()
    
    path = args.path
    sep = args.sep
    coltypesString = args.ctypes
    nminPercent = args.nmin
    nTrees = args.ntrees
    massBased = args.massbased
    
    data = readCSV(path, sep)
    nrows, ncols = data.shape
    nmin = int(np.floor(nminPercent * nrows))
    if coltypesString.endswith(','):
        coltype = int(coltypesString[0])
        coltypes = [coltype] * ncols
    else:
        coltypes = [int(c) for c in coltypesString.split(',') if c != '']
    
    from time import time
    startTime = time()
    if massBased == 0:
        matrix = getSim(data, nmin, coltypes, nTrees)
    else:
        pass
    endTime = time()
    print("Time: {:.2f}ms".format((endTime - startTime) * 1000))
    np.savetxt("matrix_uet.csv", matrix, delimiter='\t')
    
if __name__ == '__main__':
    main()
