import numpy as np
import matplotlib.pyplot as plt

img_width = 300
img_height = 300

layer_width = 3
layer_height = 3


stepy_x = img_width/layer_width
stepy_y = img_height/layer_height

linx = np.linspace(0.5*stepy_x,img_width-0.5*stepy_x,layer_width)
liny = np.linspace(0.5*stepy_y,img_height-0.5*stepy_y,layer_height)

centers_x, centers_y = np.meshgrid(linx,liny)

centers_x = centers_x.reshape(-1,1)
centers_y = centers_y.reshape(-1,1)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylim(0,300)
plt.xlim(0,300)
plt.scatter(centers_x,centers_y)
plt.show()