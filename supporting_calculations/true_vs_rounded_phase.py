#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:24:41 2025

@author: fawaz
"""

"""
Code Description:
This Python script calculates the difference between the true phase and the phase calculated after rounding the values of 
a and b (representing I and Q values) into integers. The script:

1. Defines a range of values for a and b (from -value_range to value_range) with a specified number of steps (num_steps).
2. Uses a meshgrid to generate combinations of a and b.
3. Calculates the true phase using arctan2(b, a) in degrees.
4. Rounds a and b to the nearest integers and recalculates the phase.
5. Computes the phase difference (true phase minus rounded phase).
6. Masks the phase difference where both a and b are zero (as the phase is undefined).
7. Visualizes the phase difference as a 2D contour plot, where the axes represent a and b, and the color gradient shows 
   the magnitude of the phase difference.

This implementation uses variables (value_range and num_steps) to make the range and resolution easily configurable.
"""


import numpy as np
import matplotlib.pyplot as plt

# Parameters
value_range = 10  # Maximum absolute value for a and b
num_steps = 500  # Number of steps in the range

# Define ranges for a and b values
a_values = np.linspace(-value_range, value_range, num_steps)
b_values = np.linspace(-value_range, value_range, num_steps)

# Create a meshgrid for a and b values
A, B = np.meshgrid(a_values, b_values)

# Calculate true phase
true_phase = np.arctan2(B, A) * (180 / np.pi)  # Convert to degrees

# Calculate phase after rounding a and b
A_rounded = np.round(A)
B_rounded = np.round(B)
rounded_phase = np.arctan2(B_rounded, A_rounded) * (180 / np.pi)  # Convert to degrees

# Compute the phase difference
phase_difference = true_phase - rounded_phase

# Mask invalid regions where both a and b are 0 (undefined phase)
invalid_mask = (A == 0) & (B == 0)
phase_difference[invalid_mask] = np.nan

# Plot the phase difference
plt.figure(figsize=(10, 8))
plt.contourf(A, B, phase_difference, levels=100, cmap='coolwarm', extend='both')
plt.colorbar(label='Phase Difference (degrees)')
plt.title('Phase Difference (True Phase vs Rounded Phase)')
plt.xlabel('a values')
plt.ylabel('b values')
plt.grid(True)
plt.show()