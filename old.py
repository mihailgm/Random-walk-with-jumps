import numpy as np
import numpy as np
import numpy.random as npr
from numpy import linalg as LA
import argparse

'''
Reads adjacency matrices of all graphs of the size 2 <= SIZE <= 9 
from the file adjSIZE.txt.
'''

    
'''
Returns list() of adjacency matrices of all graphs of the size 2 <= SIZE <= 9 
in numpy array format
'''     
def GetGraphsAsNumpyArrayFromFile(n):
    numpys = list()
    auxiliary_arrays = list()
    for line in adj:
        newline = list(line)
        if newline != ['\n']:
            auxiliary_arrays.append(newline[:-1])

    for i in range(len(auxiliary_arrays)):
        if auxiliary_arrays[i][0] == 'G':
            new_numpy = np.ones((n, n))
            for j in range(n):
                new_numpy[j] = auxiliary_arrays[i + j + 1]
            numpys.append(new_numpy)
    return numpys

'''
Gets ALL ergodic matrices of graphs with n vertices
''' 
def GetAllErgodicMatrices(n):
    numpys = GetGraphsAsNumpyArrayFromFile(n)
    ans = list()
    for matrix in numpys:
        if CheckIfErgodic(matrix):
            for row in matrix: row /= np.sum(row)
            ans.append(matrix)
    return ans

'''
Checks if matrix is ergodic. Works only for matrices with nonnegative entries.
Roughly speaking, we can check ergodicity of non-stochastic matrices.
''' 
def CheckIfErgodic(matrix):
    powered_matrix = LA.matrix_power(matrix, 2 * len(matrix))
    for row in powered_matrix:
        for elem in row:
            if elem == 0:
                return False
    return True

'''
Using previous functions, this one produces an ergodic stochastic matrix.
'''
def GenerateErgodic(n):
    matrix = GenerateSymmetricBoolean(n)
    while not CheckIfErgodic(matrix):
        matrix = GenerateSymmetricBoolean(n)
    for row in matrix: row /= np.sum(row)
    return matrix

'''
Extracts an eigenvalue with the second largest absolute value
'''        
def ExtractLAMBDA(eigenvalues):
    sorted_values = np.sort(eigenvalues)
    return sorted_values[0] if abs(sorted_values[0]) > abs(sorted_values[-2]) else sorted_values[-2]

'''
Returns three values:
1) The eigenvalue LAMBDA with the second smallest absolute value
2) The eigenvector, corresponding to this eigenvalue
3) Boolean value indicating if LAMBDA is negative

NOTE: eigenvector's Euclidean norm equals to 1
'''
def LAMBDA_eigenvector_and_negativity(matrix):
    eigenvalues, eigenvectors = LA.eig(matrix)
    LAMBDA = ExtractLAMBDA(eigenvalues)
    indices = np.argwhere(eigenvalues == LAMBDA)
    return LAMBDA, eigenvectors[:, indices[0]], LAMBDA < 0


'''
Checks the condition on mixing-time decrease, derived in
the artile:
((1^T * v)^2)/n < LAMBDA * (norm(v))^2                  (*)
Returns:
1) Left-hand side of the  (*)
2) Right-hand side of the (*)
3) Boolean variable indicating the answer to the question
"Have we found a counterexample?"

NOTE: eigenvector's Euclidean norm equals to 1
thus RHS = LAMBDA
'''
def Check_Avrachenkov_and_Bogdanov_condition(v, LAMBDA):
    return (np.sum(v) * np.sum(v))/len(v), LAMBDA, (np.sum(v) * np.sum(v))/len(v) > LAMBDA

'''
Generates bipartile graph in such a way that:
- Sizes of parts are n and m
- Probabilitity of an internal edge in each part is p_internal
- Probabilitity of an edge between two parts is p_external
'''
def GenerateBipartileGraph(n, m, p_internal, p_external):
    NW = GenerateSymmetricBoolean(n, p_internal)
    NE = np.random.choice(a = [1.0, 0.0], size = (n, m), p = [p_external, 1 - p_external])
    SW = NE.T
    SE = GenerateSymmetricBoolean(m, p_internal) 
    upper = np.concatenate((NW, NE), axis = 1)
    lower = np.concatenate((SW, SE), axis = 1)
    matrix = np.concatenate((upper, lower), axis = 0)
    for i in range(m + n): matrix[i][i] = 0
    return matrix
    
def GenerateErgodicBipartileGraph(n, m, p_internal, p_external):
    matrix = GenerateBipartileGraph(n, m, p_internal, p_external)
    while not CheckIfErgodic(matrix):
        matrix = GenerateBipartileGraph(n, m, p_internal, p_external)
    for row in matrix: row /= np.sum(row)
    return matrix

'''
Main part
'''
#LAMBDA, v, negativity = LAMBDA_eigenvector_and_negativity(matrix)
#LHS, RHS, negativity = Check_Avrachenkov_and_Bogdanov_condition(v, LAMBDA)


parser = argparse.ArgumentParser()
parser.add_argument('--size','-s')
parser.add_argument('--mode','-m',choices=['all','random','random_bipartite'])
args = parser.parse_args()


if args.mode == "random":
    pass
elif args.mode == "random_bipartite":
    pass
else:
    size = int(args.size)
    adj = open('adj' + str(size) + '.txt', 'r')
    matrices = GetAllErgodicMatrices(size)
    for matrix in matrices:
        LAMBDA, v, negativity = LAMBDA_eigenvector_and_negativity(matrix)
        LHS, RHS, success = Check_Avrachenkov_and_Bogdanov_condition(v, LAMBDA)
        if not negativity:
            print(LHS,'VS', RHS)
        if not negativity and success:
            raise Exception('COUNTEREXAMPLE FOUND')
        else: print('LAMBDA < 0 ?', negativity)

