import numpy as np
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

def main():
    feature_vector = np.array([-0.21086806, 0.20753932, -0.09861461, 0.37059427, 0.37000071, 0.38881699, 0.02757386, 0.02943387, 0.38455536, 0.1994711])
    label = -1
    L = 0.23074817674264758
    eta = 0.8158556292794036
    current_theta = np.array([-0.06962791, 0.19268509, 0.49111872, -0.48535835, -0.42606142, 0.4471112, 0.11423151, -0.01128072, 0.34277434, -0.06178095])
    current_theta_0 = -1.527385905600085
    print(pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0))

if __name__ == '__main__':
    main()