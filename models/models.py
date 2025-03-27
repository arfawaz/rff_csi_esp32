#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:16:46 2025

@author: fawaz
"""
###############################################################################
#imports
import torch.nn as nn
import torch
from torchvision.models import ResNet50_Weights
from torchvision import models
from transformers import ViTConfig, ViTForImageClassification, AdamW
import math

###############################################################################
###############################################################################
#Simple CNN model 

'''
This model is designed specifically to classify the data from Adam Plutos.This model has two Convolution Layers.
Input data is of shape 64by2. This model is mostly hardcoded and
only the number of classes (num_classes) can be varied.

'''
class SimpleCNN(nn.Module):
    def __init__(self,num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1),padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1),padding=(1, 0))
        self.fc1 = nn.Linear(4096, 128)  # Adjusted based on the output shape of conv layers
        self.fc2 = nn.Linear(128, num_classes)  # 4 output classes
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))

    def forward(self, x):
        #the shape of the input is : 999by2
        x = torch.relu(self.conv1(x)) #the shape after the first convolution lahyer is : 
        x = torch.relu(self.conv2(x))
        #x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

###############################################################################

# vit_model_2

'''
This model is used to do classification task on caa input data of shape 64by2
using ViT based model. We are using a built-in transformer model from huggingface
called ViTForImageClassification which takes in a ViTConfig file which contains the
details of the model like input size, number of classes, attention head etc. We 
wrap this inside the nn.module() to create the vit_model_2. In this model we configure
the ViTConfig to do tokenziation by taking each of the 64by1 columns in the whole
64by2 and embedding them. This is achieved by setting the convolutional filter
patch size as (64,1). This model is derived from the vit_model_2 in the models.py
file in the repository named "caa_authentication".
'''

class vit_model_2(nn.Module):  # Defining a custom ViT model class inheriting from nn.Module
    def __init__(self, input_dim=(64, 2), num_classes=15, hidden_size=768, 
                 num_attention_heads=12, num_hidden_layers=12, intermediate_size=3072, 
                 patch_size=(64, 1), num_channels=1):
        
        """
        Initializes the Vision Transformer (ViT) model with custom configurations.

        Args:
        - input_dim (tuple): Dimensions of the input data (height, width). Default is (64, 2).
        - num_classes (int): Number of output classes for classification. Default is 15.
        - hidden_size (int): Size of the transformer hidden layers. Default is 768.
        - num_attention_heads (int): Number of attention heads in the transformer layers. Default is 12.
        - num_hidden_layers (int): Number of transformer layers. Default is 12.
        - intermediate_size (int): Size of the intermediate feed-forward layer in the transformer. Default is 3072.
        - patch_size (tuple): Size of each patch the model processes. Default is (64, 1).
        - num_channels (int): Number of input channels. Default is 1 for grayscale data.
        """
        
        super(vit_model_2, self).__init__()  # Calls the constructor of the parent class (nn.Module)
        
        # Store the model hyperparameters
        self.input_dim = input_dim  # Input image dimensions (Height, Width)
        self.num_classes = num_classes  # Number of classification labels
        self.hidden_size = hidden_size  # Transformer hidden layer size
        self.num_attention_heads = num_attention_heads  # Number of attention heads
        self.num_hidden_layers = num_hidden_layers  # Number of transformer layers
        self.intermediate_size = intermediate_size  # Feed-forward network size
        self.patch_size = patch_size  # Patch size for dividing the input image
        self.num_channels = num_channels  # Number of channels (e.g., grayscale = 1, RGB = 3)

        # Create ViT Configuration object with the specified parameters
        self.ViTConfig = ViTConfig(
            image_size=self.input_dim,  # Specifies the input image dimensions (height, width)
            num_labels=self.num_classes,  # Number of classes in the output classification
            hidden_size=self.hidden_size,  # Size of hidden layers in the transformer
            num_attention_heads=self.num_attention_heads,  # Number of self-attention heads per transformer layer
            num_hidden_layers=self.num_hidden_layers,  # Total transformer encoder layers
            intermediate_size=self.intermediate_size,  # Size of the feed-forward layer inside each transformer block
            patch_size=self.patch_size,  # Size of image patches that will be fed to the transformer
            num_channels=self.num_channels,  # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        )

        # Initialize the Vision Transformer model for image classification using the defined configuration
        self.ViTForImageClassification = ViTForImageClassification(self.ViTConfig)
        
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
        - x (torch.Tensor): Input tensor representing an image or batch of images.

        Returns:
        - torch.Tensor: The output logits from the ViT classification model.
        """
        x = self.ViTForImageClassification(x)  # Pass input through the ViT model
        return x

###############################################################################
