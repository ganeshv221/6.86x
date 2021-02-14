import numpy as np
import random
# Part I


#pragma: coderesponse template
def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices
#pragma: coderesponse end

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    z = label*(np.matmul(current_theta.transpose(),feature_vector) + current_theta_0)
    if z <= 1:
        current_theta = (1-eta*L)*current_theta + eta*label*feature_vector
        current_theta_0 = current_theta_0 + eta*label
    else:
        current_theta = (1-eta*L)*current_theta
    return (current_theta, current_theta_0)
    raise NotImplementedError

def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Your code here
    count = 1
    updated_classifier = (np.zeros(feature_matrix.shape[1]), 0)
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            updated_classifier = pegasos_single_step_update(feature_matrix[i], labels[i], L, 1/(count**(1/2)), updated_classifier[0], updated_classifier[1])
            count += 1
    return updated_classifier
    raise NotImplementedError

def main():
    feature_matrix = np.array([
        [-0.21086806, 0.20753932, -0.09861461, 0.37059427, 0.37000071, 0.38881699, 0.02757386, 0.02943387, 0.38455536, 0.1994711],
        [-0.21086806, 0.20753932, -0.09861461, 0.37059427, 0.37000071, 0.38881699, 0.02757386, 0.02943387, 0.38455536, 0.1994711]])
    labels = np.array([-1,-1])
    T = 5
    L = 0.23074817674264758
    print(pegasos(feature_matrix, labels, T, L))

if __name__ == '__main__':
    main()