import numpy as np
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

def main():
    feature_vector = np.array([-0.2716077,-0.00399585,-0.44559459,-0.49820941,-0.07193993,-0.1679458, 0.3672857,0.29795121,0.13233302,0.27706828])
    label = 1
    current_theta = np.array([-0.05355242,0.44716158,0.16322553,-0.29793032,-0.14963504,0.05575212,-0.0423296,-0.05115716,-0.48075965,-0.00251748])
    current_theta_0 = 0.4439898466596166
    print(perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0))
if __name__ == '__main__':
    main()  