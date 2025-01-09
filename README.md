# Value Class for Autograd

## Description
This repository implements a simple **autograd** (automatic differentiation) system in Python. The core of this system is the `Value` class, which allows for the computation of gradients through backpropagation. This class supports basic mathematical operations (addition, multiplication, subtraction, etc.) and advanced functions like **tanh** and **exp**, which are essential for machine learning and neural networks.

The `Value` class keeps track of both the value and its gradient, allowing for efficient backpropagation through complex computational graphs.

## Features
- Supports basic operations: `+`, `-`, `*`, `/`, `**`.
- Supports advanced operations: `tanh`, `exp`.
- Implements automatic gradient calculation via backpropagation.
- Customizable with labels for easy tracking.
- Supports chain rule and backward propagation in computational graphs.

## Installation
To use the `Value` class in your own project, simply clone this repository and add the file containing the class to your project:

### Clone the repo
```bash
git clone https://github.com/arshjeevs/value-autograd.git
