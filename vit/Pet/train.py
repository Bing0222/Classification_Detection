import torch
import torchvision
from torch.nn import functional as F

from model import ViT
from dataload import train_dataloader, test_dataloader

from tqdm import tqdm
import os
import json


BATCH_SIZE = 8
TRAIN_LEN = 0
TEST_LEN = 0
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
NUM_WORKERS = 4
EPOCHS = 5
DEVICE = 'cuda'
SEED = 2022
SPLIT_DATA = 0.2
LR = 0.001
EMBED_DIM = 36
NUM_HEADS = 4
DEPTH = 8
DROPOUT = 0.3


net = ViT(embed_dim = EMBED_DIM, num_classes=37, num_heads = NUM_HEADS, 
            depth = DEPTH, drop_rate = DROPOUT)

torch.nn.init.xavier_normal_(net.classifier.weight)

# print('Total number of parameters:', sum(param.numel() for param in net.parameters()))

# print('Number of trainable parameters:', sum(param.numel() for param in net.parameters() if param.requires_grad))

if torch.cuda.is_available():
  print('true')
  net = net.cuda()


optimizer = torch.optim.Adam(net.parameters(), lr=LR)

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

import wandb


def metrics_wb(dataloader, model, label_name):
    model = model.eval()
    tp = 0

    my_classes = dataset_tuple
    
    correct_pred = {classname: 0 for classname in my_classes}
    total_pred = {classname: 0 for classname in my_classes}

    with torch.no_grad():
        tracker = tqdm(dataloader, total=len(dataloader))
        for batch in tracker:
            image_batch, class_idx_batch = batch
            if torch.cuda.is_available():
                image_batch = image_batch.cuda()
                #class_idx_batch = torch.nn.functional.one_hot(class_idx_batch, num_classes = 37)
                class_idx_batch = class_idx_batch.cuda()
            
            logits = model(image_batch)
            classes = torch.argmax(logits, dim=1)
            tp += (class_idx_batch == classes).sum().item()

            for label, prediction in zip(class_idx_batch.cpu(), classes.cpu()):
              if label == prediction:
                correct_pred[my_classes[label]] += 1
              total_pred[my_classes[label]] += 1

        acc = tp / len(dataloader.dataset)
        averange_acc = label_name + '_accuracy'
        wandb.log({averange_acc: acc})
        print(averange_acc, acc)

        recall_classes_dict = {}

        for classname, correct_count in correct_pred.items():
          recall = float(correct_count) / total_pred[classname]
          recall_class = label_name + '/recall_' + classname
          if label_name == 'Test': wandb.log({recall_class: recall})
          print(f'Recall for class: {classname:5s} is {recall}')
          recall_classes_dict[classname] = recall
    
    return acc, recall_classes_dict

def train(model, num_epochs, prefix):
    
    wandb.init(
    project="pets", config={
    "TRAIN_LEN": TRAIN_LEN,
    "TEST_LEN": TEST_LEN,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "IMAGE_WIDTH": IMAGE_WIDTH,
    "IMAGE_HEIGHT": IMAGE_HEIGHT,
    "NUM_WORKERS": NUM_WORKERS,
    "SPLIT_DATA": SPLIT_DATA,
    "LR": LR,
    "MODEL_DESC": MODEL_DESC,
    "BATCH_SIZE": BATCH_SIZE,
    "EMBED_DIM": EMBED_DIM,
    "NUM_HEADS": NUM_HEADS,
    "DEPTH": DEPTH,
    "DROPOUT": DROPOUT
    })

    wandb.watch(model)

    my_classes = dataset_tuple

    best_recall = {}

    best_accuracy = 0

    for class_name in my_classes:
      best_recall[class_name] = 0

    for epoch in range(num_epochs):
        print('\nEpoch', epoch)
        model = model.train()

        loss_epoch = []

        tracker = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in tracker:
            
            optimizer.zero_grad()
            
            image_batch, class_idx_batch = batch
            if torch.cuda.is_available():
                image_batch = image_batch.cuda()
                class_idx_batch = class_idx_batch.cuda()
                # Denis OHE
                #class_idx_batch = torch.nn.functional.one_hot(class_idx_batch, num_classes = 37)
                #class_idx_batch = class_idx_batch.cuda()

            logits = model(image_batch)

            
            #print(class_idx_batch)
            loss = F.cross_entropy(logits, class_idx_batch)

            
            loss.backward()

            wandb.log({'Batch_train_loss': loss})

            loss_epoch.append(loss.detach().flatten()[0])

            
            optimizer.step()

        
        train_metrics = metrics_wb(train_dataloader, model, 'Train')
        test_metrics = metrics_wb(test_dataloader, model, 'Test')

        wandb.log({'Epoch_train_loss': sum(loss_epoch)/len(loss_epoch)})

        if test_metrics[0] > best_accuracy:
          print('\nBest accuracy in Epoch №', epoch)
          print('\nAverange accuracy: ', test_metrics[0])
          
          #best_recall = test_metrics[1]
          best_accuracy = test_metrics[0]
          
          torch.save(model.state_dict(), prefix + 'model.pth')
          torch.save(optimizer.state_dict(), prefix + 'optimizer.pth')
          
    wandb.finish()



prefix = 'ViT' + '_our_' + str(EPOCHS) + 'epoch_'
MODEL_DESC = 'Use our ViT on 1 stage with 4 aug: Fliplr, Flipud, CutOut and Rot90'
train(net, EPOCHS, prefix)





# import gc
# gc.collect()
# torch.cuda.empty_cache()

# # ViT(embed_dim = EMBED_DIM, num_classes=37)
# pretrained = ViT(embed_dim = EMBED_DIM, num_classes=37, num_heads = NUM_HEADS, 
#             depth = DEPTH, drop_rate = DROPOUT)
# #torch.nn.init.xavier_normal_(pretrained.head.weight)
# pretrained.load_state_dict(torch.load('ViT_our_2stage_60epoch_model.pth'))
# pretrained.cuda()

# sd = optimizer.state_dict()
# sd['param_groups'][0]['lr'] = 0.0005
# optimizer.load_state_dict(sd)
# LR = 0.0005

# LR = 0.00452560
# optimizer = torch.optim.Adam(pretrained.parameters(), lr=LR)

# prefix = 'ViT' + '_our_3stage_' + str(EPOCHS) + 'epoch_'
# MODEL_DESC = 'Use our ViT on 3 stage with 5 aug and new LR: Fliplr, Flipud, CutOut, Grayscale and Rot90'
# train(pretrained, EPOCHS, prefix)