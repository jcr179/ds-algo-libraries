from collections import Counter

# Complete the countingSort function below.
def countingSort(arr):
    # Time O(n+k): n is number of vals to sort, k is range of vals to sort. Best if range of vals is small...
    # Memory O(k): need k array elements to count instances of each value 
    c = Counter(arr)
    vals = []
    for x in range(max(c.keys())+1):
        if x in c:
            vals.append(x)#print(c[x], end=' ')
         
    return vals
    
a = [2, 3, 4, 1, 6]
out = countingSort(a)
print(out)