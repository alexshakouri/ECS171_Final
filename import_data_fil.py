import numpy as np
array=np.load('ecs171train.npy')


b=np.zeros([50000,771]);
for i in range (0,50000):
	a=array[i+1].decode("utf-8")
	a=a.replace("NA", "0")
	b[i]=[float(s) for s in a.split(',')]
print(b[0])
print(b[49998])
np.savetxt("data.csv", b, delimiter=" ")
print(len(b))
