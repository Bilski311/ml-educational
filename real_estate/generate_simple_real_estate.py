import numpy as np

x_train = []
for size in range(1, 100):
    x_train.append([size, size*100])

x_train = np.array(x_train)
print(x_train)
np.savetxt("simple_real_estate.csv", x_train, delimiter=",", fmt='%i')
