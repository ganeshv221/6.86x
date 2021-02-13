import numpy as np
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here
    z = np.zeros(len(feature_matrix))
    hinge_loss_sum = 0.0
    for i in range(0,len(feature_matrix)):
        z[i] = labels[i]*(np.matmul(theta.transpose(),feature_matrix[i]) + theta_0)
        if z[i] > 0:
            hinge_loss_sum = hinge_loss_sum + 0
        else:
            hinge_loss_sum = hinge_loss_sum + 1 - z[i]
    return hinge_loss_sum/len(feature_matrix)
    raise NotImplementedError

def main():
    feature_matrix = np.array([[-1,0],[0,1],[1,0],[0,-1]])
    labels = np.array([1,1,-1,-1])
    theta = np.array([1,1])
    theta_0 = 0
    print(hinge_loss_full(feature_matrix, labels, theta, theta_0))

if __name__ == '__main__':
    main()