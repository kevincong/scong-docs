=============
Deep Learning
=============

Coursera
========

Neural Networks and Deep Learning
---------------------------------

* ReLU: Rectified Linear Unit. max(0, x)
* Image Recognition -> CNN; Sequenced data (Speech recognition, Machine translation) -> RNN
* Switching from Sigmoid to ReLU activation function improves performance (faster).

* Image vector: x = [R,,, G,,, B,,,] (All vertical). Dimension = width * height * 3. X = [x^(1) x^(2) ... x^(m)]. Y = [y^(1) y^(2) ... y^(m)]
* Sigmoid function: :math:`\sigma = \frac{1}{1 + e^{-z}}`.
* Loss function, L(y^, y) = - (y * logy^ + (1 - y) * log(1 - y^))
    * if y = 1, L(y^, y) = - (logy^), want y^ large.
    * if y = 0, L(y^, y) = - (log(1 - y^)), want y^ small.
* Cost function, J(w, b) = 1 / m (sum(L(y^(i), y(i))))


Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
--------------------------------------------------------------------------------------

Structuring Machine Learning Projects
-------------------------------------

Convolutional Neural Networks
-----------------------------

Sequence Models
---------------


CS489
=====

CS486
=====