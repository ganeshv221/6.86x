def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if x<=y:
        fs = x*y
    else:
        fs = x/y
    return fs
    raise NotImplementedError
