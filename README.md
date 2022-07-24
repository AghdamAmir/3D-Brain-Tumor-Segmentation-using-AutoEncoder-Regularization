# 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization

This repository contains the pytorch implementation of the paper *'3D MRI Brain Tumor Segmentation Using Autoencoder Regularization'* by *Andriy Myronenko* which **won the 1st place** in the BraTS 2018 challenge.

## Prerequisties and Run
This code has been implemented in python language using Pytorch framework. Following environements and libraries are needed to run the code:
- Python 3
- torch version 1.12.0
- torchvision 0.13.0

## Run Demo
The **model.py** file contains the implemented model to be trained. You can, however, replace this file with any desired and arbitrary implemntation of a segmentation model. <br/>
1. Download the Brain Tumour Segmentation Dataset from [this](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) link, named as **Task01_BrainTumour.tar**.<br/>
2. Extract the tar file in your desired location.<br/>
3. Organize the training and validation data and their respective labels in seperate folders.<br/>
4. Set the path to each training and validation folder in **config.py** along with other training parameters.<br/>
5. Define your custom augmentations or transforms in **transforms.py**.<br/>
6. Finally, run the **train.py** script to start the training process. The model will automatically save the best weights for the valiation set at the 
specified location in config.py.

## Model Architecture
The model follows an Encoder-Decoder architecture. An extra image reconstruction path with Variation Auto Encoder(VAE) has been incorprated to the network's architecure to boost the model in learning more semantically important 
features in the latent space.</br>

![Model Architecture](https://github.com/AghdamAmir/3D-Brain-Tumor-Segmentation-using-AutoEncoder-Regularization/blob/main/imgs/architecture.png)

## Combined Loss Function
The model utilizes a weighted combined loss consisting of three terms. The combined loss is formulated as below: 

![Combined Loss](https://github.com/AghdamAmir/3D-Brain-Tumor-Segmentation-using-AutoEncoder-Regularization/blob/main/imgs/combined_loss.png)

The dice loss is used as the segmentation loss:

![Dice Loss](https://github.com/AghdamAmir/3D-Brain-Tumor-Segmentation-using-AutoEncoder-Regularization/blob/main/imgs/dice_loss.png)

L2 distance is used as the reconstruction loss:

![L2 Loss](https://github.com/AghdamAmir/3D-Brain-Tumor-Segmentation-using-AutoEncoder-Regularization/blob/main/imgs/L2_loss.png)

And KL loss is used as the VAE branch's loss:

![KL Loss](https://github.com/AghdamAmir/3D-Brain-Tumor-Segmentation-using-AutoEncoder-Regularization/blob/main/imgs/KL_loss.png)
