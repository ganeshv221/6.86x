import numpy as np

def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    #Your code here
    s = np.linalg.norm(A+B)
    return s
    raise NotImplementedError

def main():
  A = np.array([2,1])
  B = np.array([3,1])
  s = norm(A,B)
  print(s)

if __name__ == '__main__':
  main()