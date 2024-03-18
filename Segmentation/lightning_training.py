import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_3 import *
from dice_score import *
from model import *
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pathlib import Path
from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms


import random

import wandb

wandb.login(relogin=True)

# code du modèle :
class mon_modele(LightningModule):
    def __init__(self, n_channels, n_classes):
        super(mon_modele, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        #code here
        self.unet = UNet(n_channels, n_classes)
        self.save_hyperparameters()

    def forward(self, x):
        #code here
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.3,0.3,0.4])).cuda()
        loss = ce_loss(output, y)
        output = F.softmax(output, dim=1)
        dice = multiclass_dice_coeff(output.float(), F.one_hot(y,3).permute(0, 3, 1, 2).float())
        loss +=  1 - dice
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_dice_coef', dice, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.3,0.3,0.4])).cuda()
        loss = ce_loss(output, y)
        output = F.softmax(output, dim=1)
        dice = multiclass_dice_coeff(output.float(), F.one_hot(y,3).permute(0, 3, 1, 2).float())
        loss +=  1 - dice
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice_coef', dice, prog_bar=True)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    
def main() :

  # Using deterministic execution
  seed_everything(42)


  #declarations variables fixes :
  dir_checkpoint = Path('./checkpoint_folder/')
  dir_data = Path('./dataset/Lung_dataset_256px.hdf5')


  in_channels = 1
  out_channels = 3
  batch_size = 32
  epochs = 30


  checkpoint_callback = ModelCheckpoint(
    dirpath=dir_checkpoint,
    filename="my_model-{epoch:03d}-{val_loss:.3f}.ckpt",
    save_top_k=2,
    verbose=True,
    monitor="val_loss",
    mode="min"
  )

  transformations=transforms.Compose([
            transforms.RandomRotation(degrees=20),
            transforms.ElasticTransform(alpha= 1.0),
            # transforms.Normalize(mean=0.5, std=0.5),
            # transforms.ToTensor(), 
    ])

  # 1 Create the data loader
  #take 20% of the dataset for validation
  PatientsIdTrain,PatientsIdVal = SelectPatientsTrainVal(dir_data, 0.2)
  #If classification type  = true : datagenerator for image classification
  #If classification type  = false : datagenerator for image segmentation
  train_ds = HDF5Dataset(dir_data,PatientsIdTrain,transform=transformations, classification_type=False)
  val_ds = HDF5Dataset(dir_data,PatientsIdVal,transform=None,mode='valid', classification_type=False)
  n_train = len(train_ds)
  n_val = len(val_ds)
  n_classes = 3

  # 2 - Create data loaders
  # params
  loader_params = dict(batch_size=batch_size,num_workers=4,pin_memory=True,shuffle=False)
  train_dl = DataLoader(train_ds,**loader_params)
  val_dl = DataLoader(val_ds,**loader_params)


  #call the model implemented before :
  model = mon_modele(in_channels, n_classes)

  # initialise the wandb logger and name your wandb project
  wandb_logger = WandbLogger(project='IMA4202_lung_pathology_Segmentation')

  # add your batch size to the wandb config
  wandb_logger.experiment.config["batch_size"] = batch_size
  
  # Create the trainer
  trainer = Trainer(
    max_epochs=epochs,
    callbacks=checkpoint_callback,
    gpus=1,
    logger=wandb_logger,
    gradient_clip_val=1
    )

  # Train the model
  trainer.fit(model, train_dl, val_dl)
  wandb.finish()

if __name__ == '__main__':
    main()