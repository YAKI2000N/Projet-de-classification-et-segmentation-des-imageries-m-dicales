import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import *
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
import torchvision.transforms as transforms
import torch.nn.init as init
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import random
wandb.login()
class mon_modele(LightningModule):
    def __init__(self, n_channels, n_classes):
        super(mon_modele, self).__init__()
        self.n_classes=n_classes
        self.n_channels=n_channels
        self.resnet = ResNet(block=resnet18_config.block, n_blocks=resnet18_config.n_blocks, channels=resnet18_config.channels, output_dim=n_classes)
        self.save_hyperparameters()
        self.accuracy = torchmetrics.Accuracy(task='binary')  # Initialize accuracy metric


    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)[0]
        ce_loss = nn.BCEWithLogitsLoss().cuda()
        loss = ce_loss(output, y.float())
        probabilities = torch.sigmoid(output)
        predictions = (probabilities > 0.5).float()
        acc = self.accuracy(predictions.view(-1), y.view(-1))
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)[0]
        ce_loss = nn.BCEWithLogitsLoss().cuda()
        loss = ce_loss(output, y.float())
        probabilities = torch.sigmoid(output)
        predictions = (probabilities > 0.5).float()
        acc = self.accuracy(predictions.view(-1), y.view(-1))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
def plot_confusion_matrix(labels, pred_labels, classes, save_path=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm_display.plot(values_format='d', cmap='Blues', ax=ax)
    plt.title('Confusion Matrix_res')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path : 
        plt.savefig(save_path)
    else : 
        plt.show()

def main():
    # Using deterministic execution
    seed_everything(42)

    # Déclarations variables fixes
    dir_checkpoint = Path('./checkpoint_folder/')
    dir_data = Path('./dataset/Lung_dataset_256px.hdf5')

    in_channels = 1
    out_channels = 2
    batch_size = 32
    epochs = 15

    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_checkpoint,
        filename="my_model-{epoch:03d}-{val_loss:.3f}.ckpt",
        save_top_k=2,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    transformations_train = transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transformations_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    PatientsIdTrain, PatientsIdVal = SelectPatientsTrainVal(dir_data, 0.2)

    train_ds_augmented = HDF5Dataset(dir_data, PatientsIdTrain, transform=transformations_train, classification_type=True)
    val_ds_augmented = HDF5Dataset(dir_data, PatientsIdVal, transform=transformations_val, mode='valid', classification_type=True)
    n_classes = 1

    loader_params = dict(batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)
    train_dl_augmented = DataLoader(train_ds_augmented, **loader_params)
    val_dl_augmented = DataLoader(val_ds_augmented, **loader_params)

    model = mon_modele(n_channels=in_channels, n_classes=n_classes)

    wandb_logger = WandbLogger(project='lung_pathologycd_classification_resnet')
    wandb_logger.experiment.config["batch_size"] = batch_size

    trainer = Trainer(
        max_epochs=epochs,
        callbacks=checkpoint_callback,
        gpus=1,
        logger=wandb_logger,
        gradient_clip_val=1,
    )

    trainer.fit(model, train_dl_augmented, val_dl_augmented)
    wandb.finish()

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dl_augmented:
            x, y = batch
            output = model(x)[0]
            probabilities = torch.sigmoid(output)
            predictions = (probabilities > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print("Confusion Matrix:")
    print(conf_matrix)
    classes = ['Class 0', 'Class 1']
    plot_confusion_matrix(all_labels, all_predictions, classes, save_path='confusion_matrix_res1.png')

if __name__ == '__main__':
    main()
