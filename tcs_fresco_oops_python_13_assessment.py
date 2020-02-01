import timeit
def f1():
    f= [i**2 for i in range(1,21)]
    return f
def f2():
    g = (x**2 for x in range(1,21))
    yield g
  
s1 = timeit.timeit(stmt="f1()",setup="from __main__ import f1",number=100000)
print(s1)
s2 = timeit.timeit(stmt="f2()",setup="from __main__ import f2",number=100000)
print(s2)
