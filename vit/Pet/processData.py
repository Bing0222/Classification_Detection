import os
from glob import glob
import splitfolders


dataset_classes = set()

filenames = glob('datasets/images/*.jpg')

for image in filenames:
    class_name = image.rsplit("/")[1].rsplit("\\")[1].rsplit('_', 1)[0]
    dataset_classes.add(class_name)

for dir_name in dataset_classes:
    try:
        os.makedirs('data/{}'.format(dir_name), exist_ok=True)
        print('Folder created: data/{}'.format(dir_name))
    except:
        print('Folder already exists: data/{}'.format(dir_name))

root = ""
dir_name = "datasets/images/"
for root, dirs, files in os.walk(os.path.join(root, dir_name)):
    for file in files:
        image_name = os.path.join(root, file)
        class_name = image_name.split("/")[-1].rsplit('_', 1)[0]
        new_filename = image_name.split("/")[-1].rsplit('_', 1)[1]
        new_path = os.path.join("data", class_name, new_filename)
        os.replace(image_name, new_path)



none_valid_picture = ['data/Abyssinian/34.jpg', 'data/Egyptian_Mau/167.jpg','data/Egyptian_Mau/177.jpg', 'data/Egyptian_Mau/191.jpg', 'data/Egyptian_Mau/145.jpg', 'data/Egyptian_Mau/139.jpg']

for f in none_valid_picture:
    os.remove(f)

splitfolders.ratio('data', output="datasets", seed=42, ratio=(.8, 0.2), group_prefix=None, move=False)

for name_class in dataset_classes:
  filelist = glob.glob(os.path.join("datasets/train/" + name_class + "/", "*.mat"))
  for f in filelist:
    os.remove(f)
  
  filelist = glob.glob(os.path.join("datasets/val/" + name_class + "/", "*.mat"))
  for f in filelist:
    os.remove(f)
