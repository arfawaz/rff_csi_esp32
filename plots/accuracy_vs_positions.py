# -*- coding: utf-8 -*-
"""
Created on Mon May 19 02:49:25 2025

@author: fawaz243
"""

import matplotlib.pyplot as plt  # Import the matplotlib library for plotting

# --- Data for the plot ---

# Positions from which test data is successively included
positions = ["P1", "P1-P2", "P1-P3", "P1-P4", "P1-P5", "P1-P6", "P1-P7", "P1-P8"]

# Corresponding test accuracy (%) for each position set
# These values represent test accuracies when test data is successively added from more positions
accuracy = [37.10, 43.93, 65.47, 68.90, 71.79, 72.40, 71.38, 75.29]

# --- Plotting the graph ---

# Set the figure size to make the plot larger and easier to read
plt.figure(figsize=(10, 6))

# Plot the positions vs accuracy
# - 'marker="o"' puts a dot at each data point
# - 'linestyle="-" ' draws lines connecting the points
# - 'linewidth=2' sets the thickness of the line
plt.plot(positions, accuracy, marker='o', linestyle='-', linewidth=2)

# Add a title to the plot explaining the context
plt.title("P1–P8 Range Test Accuracy (10 Common MACs)")

# Label the x-axis with position information
plt.xlabel("Position (Test data added successively)")

# Label the y-axis with accuracy
plt.ylabel("Test Accuracy (%)")

# Enable grid to make it easier to read values
plt.grid(True)

# Rotate x-axis labels slightly for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent labels from getting cut off
plt.tight_layout()

# Display the plot on the screen
plt.show()
