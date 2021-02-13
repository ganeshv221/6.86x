def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    fv = np.vectorize(scalar_function)
    return fv(x,y)
    raise NotImplementedError
