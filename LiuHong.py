#!/usr/bin/env python
# coding: utf-8

# # Plant Pathology 2020
# 
# - **v1**: Starter code
# - **v4**: 5-Folds CV. The gap between local and public score seems a bit high, probably because of the small number of validation samples. I added 5-folds (for now, simple k-fold) CV to see whether it reduces the difference. 
# - **v5**: More augmentations. I try to train for more epochs. To prevent overfitting, I added more augmentations.
# 
# |**Version**|**Net**|**#Folds**|**#Epochs**|**Local LB**|**Public LB**|**Notes**|
# |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
# |v1| Resnet-18 | 1| 10| 0.972| 0.923|Starter code|
# |v4| Resnet-18 | 5| 5| 0.950 | 0.941 |5-folds CV|
# |v5| Resnet-18 | 5| 10| n.a. | n.a. |More augmentations|

# In[ ]:

#%%
import os
import numpy as np
#%%
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

__print__ = print

def print(string):
    os.system(f'echo \"{string}\"')
    __print__(string)


# In[ ]:


DIR_INPUT = '/kaggle/input/plant-pathology-2020-fgvc7'

SEED = 42
N_FOLDS = 5
N_EPOCHS = 10
BATCH_SIZE = 64


# In[ ]:


class PlantDataset(Dataset):
    
    def __init__(self, df, transforms=None):
    
        self.df = df
        self.transforms=transforms
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        image_src = DIR_INPUT + '/images/' + self.df.loc[idx, 'image_id'] + '.jpg'
        # print(image_src)
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        labels = torch.from_numpy(labels.astype(np.int8))
        labels = labels.unsqueeze(-1)
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, labels


# In[ ]:


class PlantModel(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = torchvision.models.resnet18(pretrained=True)
        
        in_features = self.backbone.fc.in_features
        
        self.logit = nn.ModuleList(
            [nn.Linear(in_features, c) for c in num_classes]
        )
        
    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        x = F.dropout(x, 0.25, self.training)

        logit = [l(x) for l in self.logit]

        return logit


# In[ ]:


transforms_train = A.Compose([
    A.RandomResizedCrop(height=256, width=256, p=1.0),
    A.Flip(),
    A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),

    # Pixels
    A.OneOf([
        A.IAAEmboss(p=1.0),
        A.IAASharpen(p=1.0),
        A.Blur(p=1.0),
    ], p=0.5),

    # Affine
    A.OneOf([
        A.ElasticTransform(p=1.0),
        A.IAAPiecewiseAffine(p=1.0)
    ], p=0.5),

    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

transforms_valid = A.Compose([
    A.Resize(height=256, width=256, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])


# In[ ]:


submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')
submission_df.iloc[:, 1:] = 0
submission_df.head()


# In[ ]:


dataset_test = PlantDataset(df=submission_df, transforms=transforms_valid)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)


# In[ ]:


train_df = pd.read_csv(DIR_INPUT + '/train.csv')
train_labels = train_df.iloc[:, 1:].values

train_df.head()


# In[ ]:


folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros((train_df.shape[0], 4))


# In[ ]:


# Download pretrained weights.
model = PlantModel(num_classes=[1, 1, 1, 1])


# In[ ]:


def train_one_fold(model, dataloader_train, dataloader_valid):
    
    for epoch in range(N_EPOCHS):

        print('  Epoch {}/{}'.format(epoch + 1, N_EPOCHS))
        print('  ' + ('-' * 20))

        model.train()
        tr_loss = 0

        for step, batch in enumerate(dataloader_train):

            images = batch[0]
            labels = batch[1]

            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(images)

            losses = []
            for i in range(4):
                losses.append(criterion(outputs[i], labels[:, i]))

            # weights: [1.0, 1.0, 1.0, 1.0]
            loss = losses[0] + losses[1] + losses[2] + losses[3]
            loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        # Validate
        model.eval()
        val_loss = 0
        val_preds = None
        val_labels = None

        for step, batch in enumerate(dataloader_valid):

            images = batch[0]
            labels = batch[1]

            if val_labels is None:
                val_labels = labels.clone().squeeze(-1)
            else:
                val_labels = torch.cat((val_labels, labels.squeeze(-1)), dim=0)

            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            with torch.no_grad():
                outputs = model(images)

                losses = []
                for i in range(4):
                    losses.append(criterion(outputs[i], labels[:, i]))

                # weights: [1.0, 1.0, 1.0, 1.0]
                loss = losses[0] + losses[1] + losses[2] + losses[3]

                val_loss += loss.item()

                preds = torch.sigmoid(torch.stack(outputs).permute(1, 0, 2).cpu().squeeze(-1))

                if val_preds is None:
                    val_preds = preds
                else:
                    val_preds = torch.cat((val_preds, preds), dim=0)


        print('  Training Loss: {:.4f}'.format(tr_loss / len(dataloader_train)))
        print('  Validation Loss: {:.4f}'.format(val_loss / len(dataloader_valid)))
        print('  Epoch score: {:.4f}'.format(roc_auc_score(val_labels, val_preds, average='macro')))
        print('')

    return val_preds


# In[ ]:


for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df)):
    print("Fold {}/{}".format(i_fold, N_FOLDS - 1))
    print("=" * 20)
    print("")

    valid = train_df.iloc[valid_idx]
    valid.reset_index(drop=True, inplace=True)

    train = train_df.iloc[train_idx]
    train.reset_index(drop=True, inplace=True)    
    

    dataset_train = PlantDataset(df=train, transforms=transforms_train)
    dataset_valid = PlantDataset(df=valid, transforms=transforms_valid)

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    device = torch.device("cuda:0")

    model = PlantModel(num_classes=[1, 1, 1, 1])
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    plist = [{'params': model.parameters(), 'lr': 5e-5}]
    optimizer = optim.Adam(plist, lr=5e-5)
    
    val_preds = train_one_fold(model, dataloader_train, dataloader_valid)
    oof_preds[valid_idx, :] = val_preds.numpy()
    
    print("  Predicting test set...")
    print("")

    model.eval()
    test_preds = None

    for step, batch in enumerate(dataloader_test):

        images = batch[0]
        images = images.to(device, dtype=torch.float)

        with torch.no_grad():
            outputs = model(images)

            preds = torch.sigmoid(torch.stack(outputs).permute(1, 0, 2).cpu().squeeze(-1))

            if test_preds is None:
                test_preds = preds
            else:
                test_preds = torch.cat((test_preds, preds), dim=0)
                
    
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] += test_preds.numpy() / N_FOLDS

print("5-Folds CV score: {:.4f}".format(roc_auc_score(train_labels, oof_preds, average='macro')))


# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# In[ ]:


submission_df


# In[ ]:




