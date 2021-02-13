import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here
    A=np.random.random([n,1])
    return A
    raise NotImplementedError

def main():
  A = randomization(5)
  print(A)
  
if __name__ == '__main__':
  main()