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

* Image vector: :math:`x = \begin{bmatrix}R_{1} \\R_{2} \\ ... \\ R_{n} \\ G_{1} \\ G_{2} \\ ... \\ G_{n} \\ B_{1} \\ B_{2} \\ ... \\ B_{n} \end{bmatrix}`. :math:`x \in R^{height*width*3}`. :math:`x = \begin{bmatrix}. & . & . & . \\. & . & . & .\\x^{(1)} & x^{(2)} & ... & x^{(m)} \\ . & . & . & .  \\ . & . & . & .  \end{bmatrix}`. :math:`Y = \begin{bmatrix}y^{(1)} & y^{(2)} & ... & y^{(m)} \end{bmatrix}`.
* Sigmoid function: :math:`\sigma = \frac{1}{1 + e^{-z}}`.
* :math:`\hat y = \sigma(w^{T}x + b)`
* Loss function, :math:`\mathcal{L}(\hat y, y) = - (y \log\hat{y} + (1 - y) \log(1 - \hat y))`
    * if y = 1, :math:`\mathcal{L}(\hat y, y) = - (\log\hat{y})`, want :math:`\hat y` large.
    * if y = 0, :math:`\mathcal{L}(\hat y, y) = - (\log(1 - \hat{y}))`, want :math:`\hat y` small.
* Cost function,
.. math::
    \mathcal{J}(w, b) = \frac{1}{m} \sum_1^m \mathcal{L}(\hat y^{(i)}, y^{(i)}) = -\frac{1}{m} \sum_1^m \left[y^{(i)} \log\hat y^{(i)} + (1-y^{(i)}) \log(1-\hat y^{(i)})\right]
* Gradient Descent,
.. math::
    w := w - \alpha\frac{\partial \mathcal{J}(w, b)}{\partial w}
    b := b - \alpha\frac{\partial \mathcal{J}(w, b)}{\partial b}
* :math:`\frac{\text{d}\mathcal{J(a, y)}}{\text{d}a} = -\frac{a}{y} + \frac{1-y}{1-a}, (note: a = \hat y)`.
* :math:`\frac{\text{d}\mathcal{J(a, y)}}{\text{d}z} = a - y`.
* :math:`\frac{\text{d}\mathcal{J(a, y)}}{\text{d}w_1} = x_1(a - y)`.
* :math:`\frac{\text{d}\mathcal{J(a, y)}}{\text{d}w_2} = x_2(a - y)`.
* :math:`\frac{\text{d}\mathcal{J(a, y)}}{\text{d}b} = a - y`.

Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization (Coming Soon)
----------------------------------------------------------------------------------------------------

Structuring Machine Learning Projects (Coming Soon)
---------------------------------------------------

Convolutional Neural Networks (Coming Soon)
-------------------------------------------

Sequence Models (Coming Soon)
-----------------------------
