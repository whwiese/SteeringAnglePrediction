# Steering Angle Prediction for Self-Driving Cars

A pytorch implementation of NVIDIA's 2016 paper "End to End Learning for Self-Driving Cars".

---

### Test Log

#### Model 1 (01/01/2021)

New year, new model! This is my first attempt at training an end-to-end steering angle predictor. I followed the Nvidia paper specifiacations for this model, 
but added batch norm in the convolutional layers and used images of size (256,455) instead of the paper's (66,200). Training was performed on a set of about 36k images, and the validation set is 9k images from 3 separate continuous sequences in the same drive.  Here's how it went:

<img src="https://github.com/whwiese/SteeringAnglePrediction/blob/master/ModelStats/Model1/130e.png" alt="gen" width="400"/>

Optimizer = Adam, Learning Rate = 2e-5, Weight Decay = 0.01

We see the model initially get a bit worse as it starts to make predictions instead of just outputting zero, but over time it recovers and eventually performs better than just outputting zero. There is clear overfitting to the training set, but this is expected, to a certain degree, due to the different situations the driver encounters in the training and validation sets. Here's how the model performed on the training and validation sets after 130 epochs of training.

<img src="https://github.com/whwiese/SteeringAnglePrediction/blob/master/ModelStats/Model1/TrainingSet.png" alt="gen" width="400"/> <img src="https://github.com/whwiese/SteeringAnglePrediction/blob/master/ModelStats/Model1/ValidationSet.png" alt="gen" width="400"/>

Training set MSE Loss: 147.62 (vs a loss of 1030.43 if we always output zero)

Validation set MSE Loss: 410.72 (vs a loss of 632.29 if we always output zero)

We can see that the model is learning to steer, although I'm not quite ready to hook it up to my car. I'm considering improving the model by getting more training data, using stronger regularization and giving the model info on past frames so its steering decisions are influenced by speed.

---

### References

1. [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) - Bojarski et al., 2016.
2. [Driving Data](https://github.com/SullyChen/driving-datasets) - Sully Chen.
