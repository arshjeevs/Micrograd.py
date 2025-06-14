# Micrograd.py

## Description

This repository implements a minimal, educational automatic differentiation library inspired by Andrej Karpathy's micrograd. It provides the core functionality for building and training basic neural networks with gradient-based optimization.  The primary focus is on understanding the fundamental concepts of automatic differentiation rather than production-level performance.

## Features and Functionality

*   **Value Object:**  The `Value` object represents a scalar value and its associated gradient.  It tracks the operations that created it, allowing for automatic backpropagation.
*   **Basic Arithmetic Operations:** Supports addition, subtraction, multiplication, division, exponentiation, and negation on `Value` objects.
*   **Activation Functions:** Includes implementations of ReLU and sigmoid activation functions.
*   **Neural Network Building Blocks:** Provides `Neuron` and `Layer` classes for constructing feedforward neural networks.
*   **Loss Calculation:** Implements mean squared error (MSE) loss function.
*   **Backpropagation:** Automatically computes gradients using the chain rule.
*   **Optimization:** Implements a basic gradient descent optimizer.
*   **Simple Example:** Includes a basic example showcasing how to build and train a simple neural network.

## Technology Stack

*   **Python 3.x:**  The entire library is written in Python 3.

## Prerequisites

*   **Python 3.6 or higher:**  Ensure you have Python 3 installed on your system.
*   **No external dependencies:** This project is designed to be dependency-free.

## Installation Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/arshjeevs/Micrograd.py.git
    cd Micrograd.py
    ```

2.  **Optional: Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

## Usage Guide

1.  **Import the necessary modules:**

    ```python
    from micrograd.engine import Value
    from micrograd.nn import Neuron, Layer, MLP
    ```

2.  **Example: Building and training a simple neural network:**

    ```python
    import random
    from micrograd.engine import Value
    from micrograd.nn import Neuron, Layer, MLP

    # Create a multi-layer perceptron (MLP) with 2 layers, 3 inputs, and 1 output
    n = MLP(3, [4, 4, 1])

    # Prepare training data
    xs = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ]
    ys = [Value(0.0), Value(1.0), Value(0.0), Value(1.0)]  # Example target values

    # Training loop
    for k in range(20):
        # Forward pass
        ypred = [n(x) for x in xs]

        # Calculate loss (mean squared error)
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # Zero gradients
        for p in n.parameters():
            p.grad = 0.0

        # Backpropagation
        loss.backward()

        # Update parameters (gradient descent)
        for p in n.parameters():
            p.data += -0.05 * p.grad

        print(f"Iteration {k}, Loss: {loss.data}")
    ```

3.  **Explanation:**

    *   The `MLP` class is used to create a multilayer perceptron.  The first argument is the number of inputs, and the second is a list specifying the number of neurons in each layer.
    *   Training data (`xs`) and target values (`ys`) are prepared. The target values must be wrapped in `Value` objects.
    *   The training loop iterates a specified number of times.
    *   Inside the loop:
        *   The network makes predictions (`ypred`) for each input in the training data.
        *   The mean squared error loss is calculated.
        *   Gradients are zeroed before backpropagation.
        *   The `backward()` method is called on the loss to compute gradients.
        *   Parameters are updated using gradient descent.
        *   The loss is printed to monitor training progress.

## API Documentation

### `engine.py`

*   **`class Value(object)`:** Represents a scalar value and its gradient.

    *   `__init__(self, data, _children=(), _op='', label='')`: Constructor.
        *   `data`: The scalar value.
        *   `_children`: Tuple of `Value` objects that contributed to this `Value`.
        *   `_op`: The operation that created this `Value`.
        *   `label`:  Optional label for debugging.
    *   `__add__(self, other)`:  Addition.
    *   `__mul__(self, other)`:  Multiplication.
    *   `__pow__(self, other)`:  Exponentiation.
    *   `__neg__(self)`:  Negation.
    *   `__sub__(self, other)`: Subtraction.
    *   `__truediv__(self, other)`: Division.
    *   `relu(self)`:  ReLU activation function.
    *   `sigmoid(self)`: Sigmoid activation function.
    *   `backward()`:  Performs backpropagation to compute gradients.

### `nn.py`

*   **`class Neuron(object)`:** Represents a single neuron in a neural network.

    *   `__init__(self, nin)`: Constructor.
        *   `nin`:  Number of inputs to the neuron.
    *   `__call__(self, x)`:  Forward pass through the neuron.
        *   `x`:  List of input `Value` objects.
    *   `parameters(self)`:  Returns a list of the neuron's parameters (weights and bias).

*   **`class Layer(object)`:** Represents a layer of neurons in a neural network.

    *   `__init__(self, nin, nout)`: Constructor.
        *   `nin`:  Number of inputs to the layer.
        *   `nout`:  Number of neurons in the layer.
    *   `__call__(self, x)`:  Forward pass through the layer.
        *   `x`:  List of input `Value` objects.
    *   `parameters(self)`:  Returns a list of the layer's parameters.

*   **`class MLP(object)`:** Represents a multi-layer perceptron (MLP).

    *   `__init__(self, nin, nouts)`: Constructor.
        *   `nin`: Number of inputs to the MLP.
        *   `nouts`: List of integers specifying the number of neurons in each layer.
    *   `__call__(self, x)`: Forward pass through the MLP.
        *   `x`: List of input `Value` objects.
    *   `parameters(self)`: Returns a list of the MLP's parameters.

## Contributing Guidelines

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes.
4.  Write tests to cover your changes.
5.  Submit a pull request.

Please ensure your code adheres to PEP 8 style guidelines.

## License Information

This project does not currently have a specified license.  All rights are reserved by the author.

## Contact/Support Information

For questions or support, please contact the repository owner through GitHub. You can also raise an issue on the GitHub repository to report bugs or request features.