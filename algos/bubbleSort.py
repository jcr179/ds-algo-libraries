n = 3 # number of elements to sort in ascending order 
a = [1, 2, 3] # elements to sort 

def bubbleSort(a):
    # Time O(n2)
    # Memory O(1)
    # a: array of length n
    n = len(a)
    numSwaps = 0
    for i in range(n):
        numberOfSwaps = 0
        for j in range(n-1):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
                numberOfSwaps += 1
                numSwaps += 1
        if numberOfSwaps == 0:
            break
            
print("Array is sorted in %d swaps." % numSwaps)
print("First Element: %d" % a[0])
print("Last Element: %d" % a[-1])