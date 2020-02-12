def insertionSort(a):
    # Time O(n2). Could perform better than (nlogn) for small lists 
    # Memory O(1) for item being moved
    # a: array of length n 
    n = len(a)
    for i in range(1, n):
        j = i
        while j > 0 and a[j] < a[j-1]:
            a[j], a[j-1] = a[j-1], a[j]
            j -= 1
       
       
a = [2, 3, 4, 1, 6]
insertionSort(a)
print(a)