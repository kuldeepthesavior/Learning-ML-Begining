from itertools import count
def factorial():
    res = 1
    for x in count(1):
        yield res
        res = x*res

fs = factorial()
print(next(fs))
print(next(fs))
print(next(fs))
print(next(fs))
#for item in range(10):
#    print(next(fs))
