import numpy as np
import random
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

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    z = label*(np.matmul(current_theta.transpose(),feature_vector) + current_theta_0)
    if z > 0:
        theta = current_theta 
        theta_0 = current_theta_0 
    else:
        theta = current_theta + label*feature_vector
        theta_0 = current_theta_0 + label

    updated_classifier = (theta, theta_0)
    return updated_classifier
    raise NotImplementedError

def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0
    sum_theta = np.zeros(feature_matrix.shape[1])
    sum_theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            updated_classifier = perceptron_single_step_update(feature_matrix[i],labels[i],current_theta,current_theta_0)
            current_theta = updated_classifier[0]
            current_theta_0 = updated_classifier[1]
            sum_theta = sum_theta + current_theta
            sum_theta_0 = sum_theta_0 + current_theta_0
            pass
    average_classifier_full = (sum_theta/(feature_matrix.shape[0]*T), sum_theta_0/(feature_matrix.shape[0]*T))
    return average_classifier_full
    raise NotImplementedError

def main():
    feature_matrix = np.array([
        [ 0.27848303, -0.20017312,  0.35696242,  0.20131191,  0.30643464,  0.02051886, -0.08439401,  0.25188568,  0.17220102,  0.07823621],
        [-0.46324446, -0.14817478, -0.24259936,  0.02603394, -0.35161974, -0.08291647,  0.24110071,  0.20313017,  0.30910057, -0.44528563],
        [ 0.35241834, -0.28947141,  0.23848349, -0.26920291,  0.15658824, -0.36651631,  0.49690056, -0.04852961, -0.3961473 ,  0.48641705],
        [ 0.25647662,  0.01336095, -0.49454861, -0.05916431, -0.08850804,  0.2152925 , 0.3410836 ,  0.15742606, -0.42019971,  0.22560967],
        [ 0.18670195, -0.34005299, -0.351156  ,  0.27231622,  0.04722998,  0.41941308, -0.14833074,  0.17862981, -0.309936  ,  0.17139564]])
    labels = np.array([-1,1,-1,1,1])
    T = 5
    print((average_perceptron(feature_matrix, labels, T)))
if __name__ == '__main__':
    main()  