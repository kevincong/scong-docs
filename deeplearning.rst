=============
Deep Learning
=============

Coursera: Deep Learning Specialization
======================================

Neural Networks and Deep Learning
---------------------------------

* ReLU: Rectified Linear Unit. :math:`max(0, x)`.
* Image Recognition -> CNN; Sequenced data (Speech recognition, Machine translation) -> RNN
* Switching from Sigmoid to ReLU activation function improves performance (faster).
* Image vector: :math:`x = \begin{bmatrix}R_{1} R_{2} ... R_{n} G_{1} G_{2}  ...  G_{n} B_{1} B_{2} ... B_{n} \end{bmatrix}^T`. :math:`x \in R^{height*width*3}`. :math:`x = \begin{bmatrix}. & . & . & . \\. & . & . & .\\x^{(1)} & x^{(2)} & ... & x^{(m)} \\ . & . & . & .  \\ . & . & . & .  \end{bmatrix}`. :math:`Y = \begin{bmatrix}y^{(1)} & y^{(2)} & ... & y^{(m)} \end{bmatrix}`.
* Sigmoid function: :math:`\sigma = g(z) = \frac{1}{1 + e^{-z}}`. Derivative: :math:`g(z)(1 - g(z))`.
* :math:`\hat y = \sigma(w^{T}x + b)`.
* Loss function, :math:`\mathcal{L}(\hat y, y) = - (y \log\hat{y} + (1 - y) \log(1 - \hat y))`
    * if y = 1, :math:`\mathcal{L}(\hat y, y) = - (\log\hat{y})`, want :math:`\hat y` large.
    * if y = 0, :math:`\mathcal{L}(\hat y, y) = - (\log(1 - \hat{y}))`, want :math:`\hat y` small.
* Cost function,
.. math::
    \mathcal{J}(w, b) = \frac{1}{m} \sum_1^m \mathcal{L}(\hat y^{(i)}, y^{(i)}) = -\frac{1}{m} \sum_1^m \left[y^{(i)} \log\hat y^{(i)} + (1-y^{(i)}) \log(1-\hat y^{(i)})\right]
* Gradient Descent,
.. math::
    w := w - \alpha\frac{\partial \mathcal{J}(w, b)}{\partial w} \\
    b := b - \alpha\frac{\partial \mathcal{J}(w, b)}{\partial b}
* :math:`\frac{\text{d}\mathcal{J(a, y)}}{\text{d}a} = -\frac{a}{y} + \frac{1-y}{1-a}, (note: a = \hat y)`.
* :math:`\frac{\text{d}\mathcal{J(a, y)}}{\text{d}z} = a - y`.
* :math:`\frac{\text{d}\mathcal{J(a, y)}}{\text{d}w_1} = x_1(a - y)`.
* :math:`\frac{\text{d}\mathcal{J(a, y)}}{\text{d}w_2} = x_2(a - y)`.
* :math:`\frac{\text{d}\mathcal{J(a, y)}}{\text{d}b} = a - y`.
* Vectorization and Broadcasting for optimization.
* Activition functions:
    * sigmoid: Try to avoid, or for binary classification only.
    * tanh: :math:`a = \tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}`. Derivative: :math:`g\prime(z) = 1-(\tanh(z))^2`.
    * ReLU: :math:`a = max(0, z)`, Derivative: :math:`g\prime(z) = \begin{cases}0 & z < 0\\1 & z \geq 0 \end{cases}`. (Or Leaky ReLU, :math:`a = max(0.01z, z)`, derivative: :math:`g\prime(z) = \begin{cases}0.01 & z < 0\\1 & z \geq 0 \end{cases}`).
* Forward Propagation for gradient descent neural network:
.. math::
    Z^{[1]} = W^{[1]}x + b^{[1]} \\
    A^{[1]} = g^{[1]}(Z^{[1]}) \\
    Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} \\
    A^{[2]} = g^{[2]}(Z^{[2]}) = \sigma(Z^{[2]})
* Back Propagation for gradient descent neural network:
.. math::
    \mathcal{d}Z^{[2]} = A^{[2]} - Y \\
    \mathcal{d}W^{[2]} = \frac{1}{m}\mathcal{d}Z^{[2]}A^{[1]T} \\
    \mathcal{d}b^{[2]} = \frac{1}{m}*np.sum(\mathcal{d}Z^{[2]}, axis=1, keepdims=True) \\
    \\
    \mathcal{d}Z^{[1]} = W^{[2]T}\mathcal{d}Z^{[2]} \cdot g^{[1]}\prime(Z^{[1]}) \\
    \mathcal{d}W^{[1]} = \frac{1}{m}\mathcal{d}Z^{[1]}X^{T} \\
    \mathcal{d}b^{[1]} = \frac{1}{m}*np.sum(\mathcal{d}Z^{[1]}, axis=1, keepdims=True) \\


Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization (Coming Soon)
----------------------------------------------------------------------------------------------------

Structuring Machine Learning Projects (Coming Soon)
---------------------------------------------------

Convolutional Neural Networks (Coming Soon)
-------------------------------------------

Sequence Models (Coming Soon)
-----------------------------
