# infinite fibonaci generator
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b,a+b


fs=fibonacci()
print(next(fs))
print(next(fs))
print(next(fs))
print(next(fs))





