path = './hard_gt2.npy'
import numpy as np
x = np.load(path)
Yes = 0
No = 0
for each in x:
    yes = 0
    no = 0
    for val in each:
        if val==0:
            no=no+1
        else:
            yes = yes + 1
    print(yes, no)
    Yes += yes
    No += no
print(Yes, No)
print(Yes/(Yes+No))
