import cProfile
def f1():
    f= [i**2 for i in range(1,21)]
    return f
def f2():
    g = (x**2 for x in range(1,21))
    return g
  
cProfile.run('f1()')
cProfile.run('f2()')
