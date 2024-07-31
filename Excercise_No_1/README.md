# Building Machine Learning Models Using Multiple Methods

This document provides an overview of building classification machine learning models using various techniques, including **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **Decision Trees** on MNIST dataset .

## K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple and intuitive classification algorithm that classifies data points based on the labels of their nearest neighbors in the feature space. :

![KNN Visualization](E:\Python\QAI_Review_Test\KNN_plot_baseline.png)

.

## Support Vector Machine (SVM)

Support Vector Machine is a powerful classification technique that seeks to find the optimal hyperplane that best separates classes in the feature space. The SVM model was evaluated with and without PCA applied, yielding the following accuracy results:

- **Without PCA**
  - Accuracy: **74.31%**

- **With PCA**
  - Accuracy: **86.52%**

The application of PCA significantly improved the model's accuracy, demonstrating the effectiveness of dimensionality reduction in enhancing classification performance.

## Decision Trees

Decision Trees are a popular choice for classification tasks due to their interpretability and ease of use. We evaluated Decision Tree models with PCA applied at different levels of components, denoted by `k`. Here are the results:

| PCA Components (k) | Accuracy with PCA |
|--------------------|-------------------|
| 10                 | 76.19%            |
| 25                 | 77.64%            |
| 50                 | 77.20%            |
| 75                 | 77.15%            |
| 100                | 76.90%            |

The results indicate that the number of PCA components can have a significant impact on the model's performance. The highest accuracy was achieved with 25 components, suggesting an optimal balance between feature reduction and information retention.
