import os
import cv2

import torch
import torchvision
from  torch.utils.data import Dataset,DataLoader
from torchvision import transforms

import imageio
from imgaug import augmenters as iaa
import imgaug as ia 

import wandb


TRAIN_DATA_PATH = 'datasets/train/'
TEST_DATA_PATH = 'datasets/val/'

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 5
DEVICE = 'cuda'

class DatasetRAM(Dataset):
    def __init__(self,path, transform=None):
        super().__init__()

        self.path = path
        self.transform = transform

        self.data = []

        self.class_name_to_idx = {}
        self.idx_to_class_name = []

        for class_idx, class_name in enumerate(sorted(os.listdir(path))):
            self.class_name_to_idx[class_name] = class_idx
            self.idx_to_class_name.append(class_name)
            class_images_path = os.path.join(path, class_name)
            for image_name in os.listdir(class_images_path):
                image_path = os.path.join(class_images_path, image_name)

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                
                self.data.append((image, class_name))

    def __getitem__(self, idx):
        image, class_name = self.data[idx]
        
        if self.transform:
          image = self.transform(image)
        
        return image, self.class_name_to_idx[class_name]

    def __len__(self):
        return len(self.data)
    

artifact_augmentation = iaa.meta.OneOf([
        iaa.imgcorruptlike.Pixelate(severity=1),
        iaa.imgcorruptlike.JpegCompression(severity=1),
        iaa.imgcorruptlike.Brightness(severity=1),
        iaa.imgcorruptlike.MotionBlur(severity=1),
        iaa.imgcorruptlike.ShotNoise(severity=1),
        iaa.GammaContrast((0.5, 2.0)),
        iaa.meta.Identity()])

train_transform = transforms.Compose([
    iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Grayscale(alpha=(0.0, 0.8)),
        iaa.Rot90((1, 3)),
        iaa.Cutout(fill_mode="gaussian", fill_per_channel=0.5, size=0.10),
        iaa.Flipud(0.5),
        iaa.Resize({"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH})
    ]).augment_image,
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    iaa.Sequential([
        iaa.Resize({"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH})
    ]).augment_image,
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


train_dataset = DatasetRAM(TRAIN_DATA_PATH, train_transform)
test_dataset = DatasetRAM(TEST_DATA_PATH, test_transform)

TRAIN_LEN = len(train_dataset)
TEST_LEN = len(test_dataset)

print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))


train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  
    num_workers=NUM_WORKERS, 
    pin_memory=True,  
    drop_last=True  
)

test_dataloader = DataLoader(
    test_dataset,  
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=True, 
    drop_last=False  
)

net = torchvision.models.vit_b_32(pretrained=True, num_classes=37,
                                  depth=4,dropout=0.3)

print('Total number of parameters:', sum(param.numel() for param in net.parameters()))

print('Number of trainable parameters:', sum(param.numel() for param in net.parameters() if param.requires_grad))

if torch.cuda.is_available():
  print('true')
  net = net.cuda()


optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

dataset_tuple = ('British_Shorthair',
 'german_shorthaired',
 'Abyssinian',
 'yorkshire_terrier',
 'english_setter',
 'chihuahua',
 'scottish_terrier',
 'miniature_pinscher',
 'Russian_Blue',
 'english_cocker_spaniel',
 'american_bulldog',
 'pug',
 'havanese',
 'leonberger',
 'great_pyrenees',
 'american_pit_bull_terrier',
 'staffordshire_bull_terrier',
 'Sphynx',
 'Egyptian_Mau',
 'Maine_Coon',
 'pomeranian',
 'samoyed',
 'beagle',
 'newfoundland',
 'wheaten_terrier',
 'Siamese',
 'Ragdoll',
 'japanese_chin',
 'keeshond',
 'Bombay',
 'boxer',
 'Persian',
 'shiba_inu',
 'saint_bernard',
 'basset_hound',
 'Birman',
 'Bengal')