import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

#this graph shows that dog height in some cases is useful
#but in other places it is not useful 
#therefore you may need to use other relevant features 
#that will help you classify the data 

