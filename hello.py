"""#>>> # Fibonacci series:
... # the sum of two elements defines the next
... a, b = 0, 1
>>> while a < 10:
...     print(a)
...     a, b = b, a+b
"""
#programme to count the number of times value repeated in list


lis=[1,1,2,3,4,45,5,6,7,7,6,5,4,4,3,3,2,2]
print(lis)
x=int(input("number to check"))
count=0

for i  in  lis:
    if(x==i):
        count=count+1

print("total repeatation is ",count)    
        
    
