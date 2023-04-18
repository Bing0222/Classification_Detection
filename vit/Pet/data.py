import os
import cv2

import imageio
from imgaug import augmenters as iaa
import imgaug as ia 

absolute_path = 'datasets/train/'
dir_names = os.listdir(absolute_path)
counter = 0

for dir_name in dir_names:
  files_count = len([name for name in os.listdir(absolute_path + dir_name) if os.path.isfile(os.path.join(absolute_path + dir_name,name))])
  counter = counter + files_count
  print("Dir '" + dir_name + "' have " + str(files_count) + " files")

print(counter)


image_name = 'datasets/train/Abyssinian/11.jpg'
image = imageio.imread(image_name)

rotate = iaa.Resize({"height": 256, "width": 256})
        # iaa.Resize({"height": 224, "width": 224})
        #iaa.JpegCompression(compression=(62, 75)),
        #iaa.Grayscale(alpha=(0.0, 1.0)),
image_aug = rotate(image=image)

rotate = iaa.Grayscale(alpha=(0.0, 0.8))
        # iaa.Resize({"height": 224, "width": 224})
        #iaa.JpegCompression(compression=(62, 75)),
        #iaa.Grayscale(alpha=(0.0, 1.0)),
image_aug = rotate(image=image_aug)

im_rgb = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
cv2.imshow("image",im_rgb)
cv2.waitKey(0)

