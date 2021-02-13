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

def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
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
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Your code here
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            updated_classifier = perceptron_single_step_update(feature_matrix[i],labels[i],current_theta,current_theta_0)
            current_theta = updated_classifier[0]
            current_theta_0 = updated_classifier[1]
            pass
    updated_classifier_full = (current_theta, current_theta_0)
    return updated_classifier_full
    raise NotImplementedError

def main():
    feature_matrix = np.array([
        [-0.31463424, 0.33613228,-0.23989791, 0.04482276,-0.15279658, 0.23581852,-0.0312256,  0.2195246,  0.0822497, -0.4228708 ],
        [-0.20241092, 0.25319239, 0.47901399, 0.07011235,-0.23775324, 0.41253798,0.29353972,-0.16396794,-0.31287784, 0.11311752],
        [ 0.13087373, 0.0025697, -0.24438515,-0.30790784, 0.19033957,-0.00428479,0.09435942, 0.35744001, 0.25491261,-0.11485151],
        [ 0.14078768, 0.25409803, -0.23978768, 0.1413767,  0.3144793,  0.0288565, -0.409638, -0.1725350,  0.1351203,  0.09707219],
        [ 0.45108364,-0.00447346, 0.44823782 ,0.24944253, 0.39330611, 0.05969039,0.05224797, 0.30901372, 0.29163327,-0.03866428]])
    labels = np.array([-1,1,-1,1,-1])
    T = 5
    print((perceptron(feature_matrix, labels, T)))
if __name__ == '__main__':
    main()  