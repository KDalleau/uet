import numpy as np
from joblib import Parallel, delayed
import multiprocessing

def random_split(values, type):
    values = np.array(values)
    min_val = np.min(values)
    max_val = np.max(values)
    if type == 0:
        split = np.random.uniform(min_val, max_val)
    else:
        index = np.random.randint(0, len(values))
        split = values[index]
    return split

def performSplit(dataset, attributesIndices, attributes, coltypes, nodeIndices):
    randIndex = np.random.choice(len(attributesIndices))
    attributeIndex = attributesIndices.pop(randIndex)
    attribute = attributes[attributeIndex]
    type = coltypes[attribute]
    nodeIndices = np.array(nodeIndices)
    data = dataset[nodeIndices, attribute]
    if len(np.unique(data)) <= 1:
        return 1, None, None
    value = random_split(data, type)
    if type == 0:
        left_mask = data < value
    else:
        left_mask = data == value
    left = nodeIndices[left_mask]
    right = nodeIndices[~left_mask]
    return 0, left, right

def build_tree_and_update(data, nmin, coltypes, nrows, ncols):
    matrix = np.zeros((nrows, nrows))
    nodes = []
    attributes = list(range(ncols))
    attributes_indices = list(range(ncols))
    nodeIndices = np.arange(nrows)
    status, left_indices, right_indices = performSplit(
        data, attributes_indices.copy(), attributes, coltypes, nodeIndices)
    if status == 1:
        matrix[np.ix_(nodeIndices, nodeIndices)] += 1.0
        return matrix
    if len(left_indices) < nmin:
        matrix[np.ix_(left_indices, left_indices)] += 1.0
    else:
        nodes.append({'indices': left_indices, 'attributes_indices': attributes_indices.copy()})
    if len(right_indices) < nmin:
        matrix[np.ix_(right_indices, right_indices)] += 1.0
    else:
        nodes.append({'indices': right_indices, 'attributes_indices': attributes_indices.copy()})
    while nodes:
        currentNode = nodes.pop(0)
        if len(currentNode['attributes_indices']) < 1:
            indices = currentNode['indices']
            matrix[np.ix_(indices, indices)] += 1.0
            continue
        nodeIndices = currentNode['indices']
        attributes_indices = currentNode['attributes_indices']
        status, left_indices, right_indices = performSplit(
            data, attributes_indices.copy(), attributes, coltypes, nodeIndices)
        if status == 1:
            matrix[np.ix_(nodeIndices, nodeIndices)] += 1.0
        else:
            if len(left_indices) < nmin:
                matrix[np.ix_(left_indices, left_indices)] += 1.0
            else:
                nodes.append({'indices': left_indices, 'attributes_indices': attributes_indices.copy()})
            if len(right_indices) < nmin:
                matrix[np.ix_(right_indices, right_indices)] += 1.0
            else:
                nodes.append({'indices': right_indices, 'attributes_indices': attributes_indices.copy()})
    return matrix

def getSim(data, nmin, coltypes, nTrees):
    nrows, ncols = data.shape
    num_cores = multiprocessing.cpu_count()
    matrices = Parallel(n_jobs=num_cores)(delayed(build_tree_and_update)(data, nmin, coltypes, nrows, ncols) for _ in range(nTrees))
    matrix = sum(matrices) / nTrees
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
        pass  # Implement getDist function if needed
    endTime = time()
    print("Time: {:.2f}ms".format((endTime - startTime) * 1000))
    np.savetxt("matrix_uet.csv", matrix, delimiter='\t')

if __name__ == '__main__':
    main()
