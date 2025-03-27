# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:20:14 2025

@author: fawaz243
"""

# test_vit_model_2

'''
This fucntion is used to test the vit_model_2 model from models.py. This fucntion
is the exact same fucntion as that is in the utilities.py file from the "caa_authentication"
repository.
'''

def test_vit_model_2(model, test_loader, loss_fn, device):
    """
    Function to evaluate a model on the test dataset.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader providing batches of test data.
        loss_fn (torch.nn.Module): Loss function to compute the error (e.g., CrossEntropyLoss).
        device (torch.device): Device to run the testing on ('cuda' or 'cpu').

    Returns:
        tuple: (avg_loss, avg_accuracy)
            - avg_loss (float): Average loss over the test dataset.
            - avg_accuracy (float): Average accuracy over the test dataset.
    """

    # ---------------------
    # Set Model to Evaluation Mode
    # ---------------------
    # In evaluation mode, dropout and batch normalization layers behave differently.
    # - Dropout layers are disabled (all neurons are active).
    # - Batch normalization uses running averages instead of batch statistics.
    model.eval()

    # Initialize variables to track total loss, correct predictions, and total samples
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # ---------------------
    # Disable Gradient Calculation
    # ---------------------
    # `torch.no_grad()` prevents PyTorch from calculating and storing gradients.
    # - Reduces memory consumption and speeds up computation.
    with torch.no_grad():
        # Loop over each batch of data from the test_loader
        for images, labels in test_loader:
            # Move input data and labels to the specified device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # ---------------------
            # Forward Pass
            # ---------------------
            # Pass the input data through the model
            # `outputs` contains the raw model outputs (logits) before softmax activation
            outputs = model(images).logits
            
            # Compute the loss between predicted and true labels
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # ---------------------
            # Compute Accuracy
            # ---------------------
            # `torch.max(outputs, 1)` returns:
            #   - Values: highest logit value along dimension 1 (not used here)
            #   - Indices: index of the highest value along dimension 1 (predicted class)
            _, predicted = torch.max(outputs, 1)  

            # Count the number of correct predictions
            total_correct += (predicted == labels).sum().item()

            # Track the total number of samples processed so far
            total_samples += labels.size(0)

    # ---------------------
    # Calculate Average Loss and Accuracy
    # ---------------------
    # Average loss = total loss across all batches divided by number of batches
    avg_loss = total_loss / len(test_loader)

    # Average accuracy = total correct predictions / total samples
    avg_accuracy = total_correct / total_samples * 100

    # ---------------------
    # Return Results
    # ---------------------
    return avg_loss, avg_accuracy