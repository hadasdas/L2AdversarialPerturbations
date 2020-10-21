# L2AdversarialPerturbations

This code follows the paper "Most ReLU Networks Suffer from l2 Adversarial Perturbations". It demonstrates its thesis by implementing 
neural networks of varied depths, that are trained to distinguish between odd and even digits in the Mnist data set. After the 
networks are trained (all around 98% accuracy) a thousand random examples are drawn from the test set and an adversarial example is
found using Gradient Descent on the input (not the weights) with small steps. The results of this code show that the greater part of
the adversarial examples found were of small l2 distance from the drawn test examples, and so was the average distance.

In depth description:
Preprocessing: Each example is centered (that is, we have added a constant to each image, so that the average value of the pixels is 0), 
and then normalized to have a squared norm of 784 (the number of pixels).
Network training:  We have trained a fully connected network, with 100 neurons in each layer, starting from standard Xavier initialization. 
We have used networks of depths 2-8.
Adversarial Examples: For each trained network, we have sampled 1000 examples, and for each example we have used gradient descent with 
the step size of 0.05, found an adversarial example, and calculated its distance from the initial example. 
We have reported the average distance for each depth and plotted a histogram to demonstrate the scattering of the distances.
