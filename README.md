# Handwritten Digit Recognition System using MNIST 
## got validation accuracy 98%

![image](https://github.com/SaadElDine/Handwritten-Digit-Recognition-MNIST/assets/113860522/ea290e25-80a4-4cf4-88ab-6a675125607b)

## Machine Learning

In the vast landscape of deep learning and artificial intelligence, the ability to recognize handwritten digits is a fundamental yet challenging task. This project develops a system that recognizes handwritten digits, leveraging the power of deep learning techniques. The focal point is the creation of a PyTorch-based neural network model trained on the widely acclaimed MNIST dataset, containing images of handwritten digits ranging from 0 to 9.

### Implementation Steps:

1. Loading the training dataset from a CSV file.
2. Moving the "label" column to the last column for consistency.
3. Separating features (pixels) and labels for further processing.
4. Defining a custom dataset class (`CustomDataset`) that inherits from PyTorch's `Dataset` class.
5. Converting Pandas DataFrames to PyTorch tensors for compatibility.
6. Defining a transformation using PyTorch's `Compose` class, which includes normalization.
7. Creating instances of the `CustomDataset` class for the training dataset.
8. Splitting the training dataset into training and validation sets using `train_test_split` from sklearn.
9. Defining the batch size and creating data loaders for training and validation sets using PyTorch's `DataLoader` class.
10. Defining a function (`visualize_samples`) for visualizing sample images and their labels.

### Architecture Dropout and Layer Normalization

- Defined a custom neural network class (`Model`) that inherits from PyTorch's `nn.Module`.
- Initializing layers, including fully connected (linear) layers, dropout layers, and layer normalization.
- Utilizing He initialization for weights.
- Specifying the loss function (`CrossEntropyLoss`) and optimizer (Stochastic Gradient Descent - SGD).
- Implementing the forward pass of the neural network using ReLU activation functions and SoftMax for multiclass classification.
- Applying layer normalization and dropout for regularization.

### Training and Validation Functions:

- `fit`: Method for training the model. Computes loss, performs backward pass, and updates weights using the optimizer.
- `predict`: Method for making predictions. Returns the index of the maximum value in the output tensor.
- `train` function for training the neural network.
- `validate` function for evaluating the model on the validation set.
- Proper use of `model.train()` and `model.eval()` to ensure correct behavior of layers like dropout during training and evaluation.

### Training Loop:

- Default training loop for a specified number of epochs.
- Exploration of different hyperparameters such as learning rates and batch sizes.
- Visualizations to understand the effects of hyperparameter choices.

  ![image](https://github.com/SaadElDine/Handwritten-Digit-Recognition-MNIST/assets/113860522/a960e3cd-be2d-4ab9-9c70-6cea4c845ee5)


### Hyperparameter Tuning:

- Exploring the impact of different learning rates on training and validation performance.
- Exploring the impact of different batch sizes on training and validation performance.
- Grid search over specified hyperparameters to find the best combination.

  ![image](https://github.com/SaadElDine/Handwritten-Digit-Recognition-MNIST/assets/113860522/23396e22-f5e8-4c66-8ce5-cf111f61092c)


### Model Evaluation:

- Loading and preparing the test data.
- Utilizing the `evaluate` function to assess the model's performance on the test set.
- Visualization of 5 samples from the test set without shuffling.
- Function (`visualize_samples_with_predictions`) to visualize model predictions for a subset of test samples.

### Conclusion:

In conclusion, the exploration into the realm of deep learning unfolded with a focus on handwritten digit recognition. The creation of a custom PyTorch model, the systematic experimentation with hyperparameters, and the investigation into the subtleties of training dynamics offered valuable insights. What enriched the model's robustness is the incorporation of dropout layers and layer normalization. This project not only resulted in the development of an effective digit recognition system but also highlighted the iterative and dynamic nature of machine learning endeavors.
