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

def classifier(feature_matrix, labels, T, L):
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

def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    labels = np.zeros(feature_matrix.shape[0])
    for k in range(0, feature_matrix.shape[0]):   
        if np.matmul(theta.transpose(),feature_matrix[k]) + theta_0 > 0:
            labels[k] = 1
        else:
            labels[k] = -1
    return labels
    raise NotImplementedError

def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Your code here
    trained_classifier = classifier(train_feature_matrix, train_labels, **kwargs)
    return (accuracy(classify(train_feature_matrix, trained_classifier[0], trained_classifier[1]), train_labels),
            accuracy(classify(val_feature_matrix, trained_classifier[0], trained_classifier[1]), val_labels))
    raise NotImplementedError

def main():
    train_feature_matrix = np.array([
        [ 0.27848303, -0.20017312,  0.35696242,  0.20131191,  0.30643464,  0.02051886, -0.08439401,  0.25188568,  0.17220102,  0.07823621],
        [-0.46324446, -0.14817478, -0.24259936,  0.02603394, -0.35161974, -0.08291647,  0.24110071,  0.20313017,  0.30910057, -0.44528563],
        [ 0.35241834, -0.28947141,  0.23848349, -0.26920291,  0.15658824, -0.36651631,  0.49690056, -0.04852961, -0.3961473 ,  0.48641705],
        [ 0.25647662,  0.01336095, -0.49454861, -0.05916431, -0.08850804,  0.2152925 , 0.3410836 ,  0.15742606, -0.42019971,  0.22560967],
        [ 0.18670195, -0.34005299, -0.351156  ,  0.27231622,  0.04722998,  0.41941308, -0.14833074,  0.17862981, -0.309936  ,  0.17139564]])
    val_feature_matrix = np.array([
        [ 0.27848303, -0.20017312,  0.35696242,  0.20131191,  0.30643464,  0.02051886, -0.08439401,  0.25188568,  0.17220102,  0.07823621],
        [-0.46324446, -0.14817478, -0.24259936,  0.02603394, -0.35161974, -0.08291647,  0.24110071,  0.20313017,  0.30910057, -0.44528563],
        [ 0.35241834, -0.28947141,  0.23848349, -0.26920291,  0.15658824, -0.36651631,  0.49690056, -0.04852961, -0.3961473 ,  0.48641705],
        [ 0.25647662,  0.01336095, -0.49454861, -0.05916431, -0.08850804,  0.2152925 , 0.3410836 ,  0.15742606, -0.42019971,  0.22560967],
        [ 0.18670195, -0.34005299, -0.351156  ,  0.27231622,  0.04722998,  0.41941308, -0.14833074,  0.17862981, -0.309936  ,  0.17139564]])
    T = 5
    L = 0.23074817674264758
    train_labels = np.array([-1,1,-1,1,1])
    val_labels = np.array([-1,1,-1,1,1])
    print(classifier_accuracy(classifier, train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L))

if __name__ == '__main__':
    main()