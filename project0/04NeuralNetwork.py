import numpy as np
def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    out=np.tanh(np.matmul(weights.transpose(),inputs))
    return out
    raise NotImplementedError

def main():
  inputs = np.array([1,2])
  weights = np.array([3,4])
  out = neural_network(inputs, weights)
  print(out)

if __name__ == '__main__':
  main()