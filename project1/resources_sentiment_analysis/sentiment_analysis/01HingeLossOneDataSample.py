import numpy as np
def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code here
    z = label*(np.matmul(theta.transpose(),feature_vector) + theta_0)
    if z >0:
        hinge_loss = 0
    else:
        hinge_loss = 1 - z
    return hinge_loss
    raise NotImplementedError

def main():
    feature_vector = np.array([-1,0])
    label = 1
    theta = np.array([1,1])
    theta_0 = 0
    hinge_loss = hinge_loss_single(feature_vector, label, theta, theta_0)
    print(hinge_loss)

if __name__ == '__main__':
    main()
