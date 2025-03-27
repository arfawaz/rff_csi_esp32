# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:51:17 2025

@author: fawaz243
"""
'''
The CustomDataset_vit_model_2 class is used to create the dataset class for the
csi data for vit_model_2. This is the same dataset class in the utilities.py file
in the "caa_authentication" repository.
'''
# CustomDataset_vit_model_2

class CustomDataset_vit_model_2(Dataset):
    def __init__(self, data, labels):
        """
        Initialize the CustomDataset class.
        
        Args:
            data (torch.Tensor): A tensor containing the input data of shape (N, 64, 2),
                                 where:
                                 - N = number of samples
                                 - 64 = sequence length (or time steps)
                                 - 2 = number of features (or channels)
            labels (torch.Tensor): A tensor containing the class labels of shape (N,),
                                   where N = number of samples.
        
        Example:
            data shape: (10000, 64, 2)
            labels shape: (10000,)
        """
        self.data = data      # Store the input data
        self.labels = labels  # Store the labels
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        This allows the DataLoader to know how many samples are available.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)  # Return the length of the dataset
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample and its corresponding label based on the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            torch.Tensor: A data sample of shape (1, 64, 2), where:
                          - 1 = channel dimension (for compatibility with CNN/Transformer models)
                          - 64 = sequence length
                          - 2 = number of features (or channels)
            torch.Tensor: Corresponding label (scalar value).
        
        Example:
            If the original sample shape is (64, 2), it is reshaped to (1, 64, 2)
            using `unsqueeze(0)`, which adds a channel dimension.
        """
        # Extract the sample at the specified index
        sample = self.data[idx]  # Shape: (64, 2)
        
        # Add a channel dimension at the beginning to make it compatible with CNN/Transformer models
        sample = sample.unsqueeze(0)  # Shape becomes: (1, 64, 2)
        
        # Extract the corresponding label
        label = self.labels[idx]
        
        return sample, label

