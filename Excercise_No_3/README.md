# Building a Triplet Loss CNN Model

## Overview

Training a Convolutional Neural Network (CNN) with triplet loss involves optimizing the model to learn embeddings that maintain a specified margin between similar and dissimilar items. Here’s a summary of model performance with varying batch sizes and training times.

## Performance and Training Time

| Batch Size | Accuracy | Time (s) |
|-------------|----------|----------|
| 8           | 88.00%   | 1360     |
| 16          | 100.00%  | 1040     |
| 32          | 100.00%  | 820      |
| 64          | 100.00%  | 560      |
| 128         | 100.00%  | 340      |
| 256         | 100.00%  | 286      |

## Key Observations

- **Accuracy**: Achieves 100% accuracy with batch sizes of 16 or greater.
- **Training Time**: Training time decreases significantly with larger batch sizes, from 1360 seconds with batch size 8 to 176 seconds with batch size 256.

## Summary

Using larger batch sizes for training a CNN with triplet loss improves accuracy and reduces training time, making the process more efficient as batch size increases.
