import numpy as np
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

def main():
    feature_matrix = np.array([
        [ 0.27848303, -0.20017312,  0.35696242,  0.20131191,  0.30643464,  0.02051886, -0.08439401,  0.25188568,  0.17220102,  0.07823621],
        [-0.46324446, -0.14817478, -0.24259936,  0.02603394, -0.35161974, -0.08291647,  0.24110071,  0.20313017,  0.30910057, -0.44528563],
        [ 0.35241834, -0.28947141,  0.23848349, -0.26920291,  0.15658824, -0.36651631,  0.49690056, -0.04852961, -0.3961473 ,  0.48641705],
        [ 0.25647662,  0.01336095, -0.49454861, -0.05916431, -0.08850804,  0.2152925 , 0.3410836 ,  0.15742606, -0.42019971,  0.22560967],
        [ 0.18670195, -0.34005299, -0.351156  ,  0.27231622,  0.04722998,  0.41941308, -0.14833074,  0.17862981, -0.309936  ,  0.17139564]])
    theta = np.array([-0.10620078,  0.30229792, -0.71325016,  0.21240517, -0.24155596,
        0.57319711, -0.1694603 ,  0.19965863, -0.00724442, -0.26983177])
    theta_0 = -0.04
    print((classify(feature_matrix, theta, theta_0)))
if __name__ == '__main__':
    main()  
