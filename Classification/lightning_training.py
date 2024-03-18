import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg import *
from dataset_3 import * 
from dice_score import *
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pathlib import Path
import torchmetrics
from torchmetrics.functional import precision
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import torch.nn.init as init
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import random
wandb.login()


# Add a function to check labels
def check_labels(dataset):
    unique_labels = set()
    for _, y in dataset:
        unique_labels.update(y.numpy().tolist())
    return unique_labels
    
# code du modèle : 
class mon_modele(LightningModule):
    def __init__(self, n_channels, n_classes):
        super(mon_modele, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.vgg16=VGG16(n_channels, n_classes)
        self.apply(self._initialize_weights)
        self.save_hyperparameters()
        self.accuracy = torchmetrics.Accuracy(task='binary')  # Initialize accuracy metric
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.vgg16(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        ce_loss = nn.BCEWithLogitsLoss().cuda()
        loss = ce_loss(output, y.float()) # Ajustement ici
        # Apply sigmoid to the output to get probabilities
        probabilities = torch.sigmoid(output)
        # Round probabilities to binary values (0 or 1) based on a threshold of 0.5
        predictions = (probabilities > 0.5).float()
        # Calculate accuracy
        acc = self.accuracy(predictions.view(-1), y.view(-1))
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        ce_loss = nn.BCEWithLogitsLoss().cuda()
        loss = ce_loss(output, y.float())  # Ajustement ici
        # Apply sigmoid to the output to get probabilities
        probabilities = torch.sigmoid(output)
        # Round probabilities to binary values (0 or 1) based on a threshold of 0.5
        predictions = (probabilities > 0.5).float()
        acc = self.accuracy(predictions.view(-1), y.view(-1))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

        
       

def plot_confusion_matrix(labels, pred_labels, classes, save_path=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm_display.plot(values_format='d', cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path : 
        plt.savefig(save_path)
    else : 
        plt.show()

def visualize_predictions(model, dataloader, num_images=10, save_path=None):
    model.eval()
    images = []
    true_labels = []
    predictions = []
    losses = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_images:
                break

            x, y = batch
            output = model(x)
            probabilities = torch.sigmoid(output)
            predicted_labels = (probabilities > 0.5).float()

            loss = nn.BCEWithLogitsLoss()(output, y.float()).item()

            images.append(x.squeeze().cpu().numpy())
            true_labels.append(y.cpu().numpy())
            predictions.append(predicted_labels.cpu().numpy())
            losses.append(loss)

    images = np.array(images)
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    losses = np.array(losses)

    classes = ['Class 0', 'Class 1']

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            axes[i, j].imshow(images[idx, 0], cmap='gray')
            axes[i, j].set_title(f'True Label: {classes[int(true_labels[idx][0].item())]}, Prediction: {classes[int(predictions[idx][0].item())]}', fontsize=8)
            axes[i, j].text(0.5, 1.1, f'Loss: {losses[idx]:.4f}', ha='center', va='center', transform=axes[i, j].transAxes, fontsize=8, color='red')
            axes[i, j].axis('off')

    # Ajustement de l'espacement entre les sous-graphiques
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()





   
def main() : 

  # Using deterministic execution ss
  seed_everything(42)

  
  #declarations variables fixes :
  dir_checkpoint = Path('./checkpoint_folder/')
  dir_data = Path('./dataset/Lung_dataset_256px.hdf5')

  
  in_channels=1
  out_channels=2
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
  

  # 1 Create the data loader
  #take 20% of the dataset for validation
  transformations=transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),

    ])
  PatientsIdTrain,PatientsIdVal = SelectPatientsTrainVal(dir_data, 0.2)
  #If classification type  = true : datagenerator for image classification
  #If classification type  = false : datagenerator for image segmentation 
  #train_ds = HDF5Dataset(dir_data,PatientsIdTrain,transform=None, classification_type=True) 
  #val_ds = HDF5Dataset(dir_data,PatientsIdVal,transform=None,mode='valid', classification_type=True)
  train_ds_augmented = HDF5Dataset(dir_data, PatientsIdTrain, transform=transformations, classification_type=True)
  val_ds_augmented = HDF5Dataset(dir_data, PatientsIdVal, transform=None, mode='valid', classification_type=True) 

  # Check the labels in the datasets
  print("Unique labels in training dataset:", check_labels(train_ds_augmented))
  print("Unique labels in validation dataset:", check_labels(val_ds_augmented))
  n_train = len(train_ds_augmented)
  n_val = len(val_ds_augmented)
  n_classes = 1
    
  # 2 - Create data loaders
  # params
  loader_params = dict(batch_size=batch_size,num_workers=4,pin_memory=True,shuffle=False) 
  #train_dl = DataLoader(train_ds,**loader_params)
  #val_dl = DataLoader(val_ds,**loader_params)
  train_dl_augmented = DataLoader(train_ds_augmented, **loader_params)
  val_dl_augmented = DataLoader(val_ds_augmented, **loader_params)


  
  #call the model implemented before :
  model = mon_modele(n_channels=in_channels, n_classes=n_classes)
  
  # Initialisation de wandb
  wandb_logger = WandbLogger(project='lung_classification')

  # add your batch size to the wandb config
  wandb_logger.experiment.config["batch_size"] = batch_size

  # Créer le Trainer
  trainer = Trainer(
    max_epochs=epochs,
    callbacks=[checkpoint_callback],
    gpus=1,
    logger=wandb_logger,
    gradient_clip_val=1,
  )


  # Train the model
  trainer.fit(model, train_dl_augmented, val_dl_augmented)
  wandb.finish()

  #vusaliser les résultats des prédictions
  visualize_predictions(model, val_dl_augmented, num_images=12, save_path='resultsLR.png')

  model.eval()
  all_predictions = []
  all_labels = []
  with torch.no_grad():
    for batch in val_dl_augmented:
        x, y = batch
        output = model(x)
        probabilities = torch.sigmoid(output)
        predictions = (probabilities > 0.5).float()
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
  all_predictions = np.array(all_predictions)
  all_labels = np.array(all_labels)  
  classes = ['Class 0', 'Class 1']
  plot_confusion_matrix(all_labels, all_predictions, classes, save_path='confusion_matrixlr.png')
  
 
 

if __name__ == '__main__':
    main() 