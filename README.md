# Coursera IBM AI Capstone Project with Deep Learning

Completed by Martti Nirkko on 23rd August, 2023.

This repository contains all the notebooks and scripts used to complete the final Capstone project of the Coursera module **IBM AI Engineering**.

> Learners who have completed this 6 course Professional Certificate have a practical understanding of Machine Learning (ML) & Deep Learning (DL). They have technical skills to start a career in AI Engineering, and can:
>
> • Implement ML algorithms including Classification, Regression, Clustering, and Dimensional Reduction using scipy & scikitlearn
> 
> • Perform ML on Big Data and deploy ML Algorithms and Pipelines on Apache Spark
> 
> • Demonstrate understanding of Deep Learning models such as autoencoders, restricted Boltzmann machines, convolutional networks, recursive neural networks, and recurrent networks
> 
> • Build deep learning models and neural networks using Keras, PyTorch and Tensorflow libraries
> 
> • Demonstrate ability to present and communicate outcomes of deep learning projects
> 

**Topics covered**: Machine Learning, Neural Networks, Deep Learning, Computer Vision, Image Processing, Scaling

**Programming languages used**: Python

**Technical skills acquired**: Classification methods (SciPy, Scikit-Learn), Deep Neural Networks (Keras, PyTorch, Tensorflow), Image Classification and Object Detection (Pillow, OpenCV), AI model deployment (IBM Watson Studio), GPU computing (Google Colab)


## Executive summary

Crack detection has vital importance for structural health monitoring and inspection.
In this project, I built deep learning models to obtain a classifier for detecting cracks in images of concrete (see Figure 1).
This was done using the PyTorch and Keras modules in Python.
While it is relatively straightforward to write neural networks with many hidden layers, training them is computationally very expensive.
The process can be sped up significantly with GPU computing.
Using various pre-trained models both with PyTorch and Keras, a classification accuracy of more than 99% was achieved.

![image](https://github.com/mnirkko/deeplearning/assets/6942556/4df2c4ff-7564-4d7c-a431-c9325dd85509)

**Figure 1** -- Images of concrete from validation dataset. Left: without cracks (negative). Right: with cracks (positive).

## Methodology

A dataset of 40'000 images of concrete was provided by the course.
The data is labeled as Positive (1) for images of concrete containing cracks, and Negative (0) when no cracks are visible.
Images are resized to 224x224 pixels to input them into the pre-trained models.

* In PyTorch, the dataset consisted of 30'000 training samples and 10'000 validation samples.
* In Keras, the dataset consisted of 30'000 training samples, 9'500 validation samples and 500 test samples.

The cross-entropy loss function was used as criterion for training, and the ResNet18 model was trained in batches of 100 images using the Adam optimiser (an extended version of stochastic gradient descent).
This process is very slow, and took around 3 hours to complete.
Over many iterations, the model improves by minimising the loss function (see Figure 2).

![image](https://github.com/mnirkko/deeplearning/assets/6942556/ce184df5-3a2b-40f8-84c2-fd86541525ab)

**Figure 2** -- Loss function of pre-trained ResNet18 model used in PyTorch. Cross-entropy loss was used as criterion.

Loading pre-trained image classification models such as ResNet50 and VGG16 allows for speeding up the process.
A sequential Dense layer is added to the output of the pre-trained models to reduce the number of output parameters to a binary value (positive/negative), which allows us to classify the images.

![image](https://github.com/mnirkko/deeplearning/assets/6942556/f788018f-8843-4022-b006-07f09fa2e514)

**Figure 3** -- Summary of pre-trained models used in Keras. Left: ResNet50 model. Right: VGG16 model. A Dense layer was added sequentially to reduce the output parameters to 2, corresponding to a binary classification. 

After processing all the batches in the training dataset, the model is evaluated using the validation dataset.
This allows us to determine the loss and accuracy of the model in a large number of iterations, as the data can be split into batches randomly.
The entire process can be repeated a given number of times, each of which is known as an epoch.
In this example, the number of epochs was chosen to be 1.
Even with pre-trained models, GPU computing is very strongly recommended.
I achieved this using Google Colab, which reduced the training time of one specific epoch from 3 hours to 2 minutes.
Finally, one can compare the resulting accuracy of each model to select the best one.

## Results

The following results were obtained:
* Using the **ResNet18** model in PyTorch, the achieved accuracy  was **99.6%**.
* Using the **VGG16** model in Keras, the achieved accuracy  was **99.6%**.
* Using the **ResNet50** model in Keras, the achieved accuracy  was **100%**.

Given the relatively simple nature of the problem (a crack should be clearly visible in a close-up image of concrete), the trained models performed very well on the validation datasets.
Thus, the predictions on the test images were also found to be accurate.
