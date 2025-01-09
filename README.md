
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
git clone https://github.com/yourusername/value-autograd.git
```

### Usage

1. Import the `Value` class from the repository:
   ```python
   from value_autograd import Value
   ```

2. Create instances of `Value` to represent variables:
   ```python
   a = Value(2.0, label="a")
   b = Value(3.0, label="b")
   c = a * b + a ** 2
   ```

3. Perform backward propagation to compute gradients:
   ```python
   c.backward()
   print(a.grad)  # Gradient of `a`
   print(b.grad)  # Gradient of `b`
   ```

## Example
Here is a simple example of how to use the `Value` class to compute gradients:

```python
import math
from value_autograd import Value

# Create values
a = Value(2.0, label="a")
b = Value(3.0, label="b")

# Define operations
c = a * b + a ** 2

# Perform backward pass
c.backward()

# Output gradients
print(f"a.grad = {a.grad}")
print(f"b.grad = {b.grad}")
```

Output:
```text
a.grad = 7.0
b.grad = 2.0
```

In this example, `a.grad` is computed as the derivative of `c` with respect to `a`, and `b.grad` is computed as the derivative of `c` with respect to `b`.

## Backpropagation Explanation
The `backward()` function propagates gradients backwards through the computational graph in reverse topological order. This ensures that each value’s gradient is computed based on the chain rule of derivatives.

For example, given the operation `c = a * b + a ** 2`, the gradients are computed as follows:
- The gradient of `c` with respect to `a` is `b + 2 * a`.
- The gradient of `c` with respect to `b` is `a`.

## Contributing
Feel free to contribute to this project by forking the repository and submitting pull requests. Contributions can include:
- New mathematical operations
- Bug fixes
- Performance improvements
- Documentation improvements

### Steps to contribute:
1. Fork the repository.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/value-autograd.git
   ```
3. Create a new branch for your changes:
   ```bash
   git checkout -b feature-name
   ```
4. Make your changes and commit them:
   ```bash
   git commit -m "Add new feature"
   ```
5. Push your changes to your fork:
   ```bash
   git push origin feature-name
   ```
6. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
