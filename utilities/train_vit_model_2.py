# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:10:44 2025

@author: fawaz243
"""
# train_vit_model_2

'''
This fucntion is used to train the vit_model_2 model from models.py. This fucntion
is the exact same fucntion as that is in the utilities.py file from the "caa_authentication"
repository.
'''


def train_vit_model_2(model, train_loader, optimizer, loss_fn, device):
    """
    Function to train a model for one epoch.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader providing batches of training data.
        optimizer (torch.optim.Optimizer): Optimizer to update model weights (e.g., Adam, SGD).
        loss_fn (torch.nn.Module): Loss function to compute the error (e.g., CrossEntropyLoss).
        device (torch.device): Device to run the training on ('cuda' or 'cpu').

    Returns:
        tuple: (avg_loss, avg_accuracy)
            - avg_loss (float): Average loss over the training dataset.
            - avg_accuracy (float): Average accuracy over the training dataset.
    """

    # Set the model to training mode
    # This enables certain layers like dropout and batch normalization to behave differently during training.
    model.train()

    # Initialize variables to track the total loss, correct predictions, and total samples
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Loop over each batch of data provided by the train_loader
    for images, labels in train_loader:
        # Move input data and labels to the specified device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # Zero out the gradients from the previous step to prevent accumulation
        optimizer.zero_grad()

        # ---------------------
        # Forward Pass
        # ---------------------
        # Pass the input data through the model
        # `outputs` contains the raw model outputs (logits) before softmax activation
        outputs = model(images).logits  

        # Compute the loss between predicted and true labels
        loss = loss_fn(outputs, labels)

        # Add current batch loss to the total loss (for calculating average loss later)
        total_loss += loss.item()

        # ---------------------
        # Backward Pass and Optimization
        # ---------------------
        # Compute gradients by backpropagation
        loss.backward()

        # Update model parameters using the optimizer
        optimizer.step()

        # ---------------------
        # Compute Accuracy
        # ---------------------
        # Get the index of the maximum logit value along dimension 1 (class prediction)
        _, predicted = torch.max(outputs, 1)  # Shape of predicted = [batch_size]

        # Count the number of correct predictions
        total_correct += (predicted == labels).sum().item()

        # Track the total number of samples processed so far
        total_samples += labels.size(0)

    # Compute the average loss over the entire training set
    avg_loss = total_loss / len(train_loader)

    # Compute the average accuracy over the entire training set
    avg_accuracy = total_correct / total_samples * 100

    # Return the average loss and accuracy
    return avg_loss, avg_accuracy