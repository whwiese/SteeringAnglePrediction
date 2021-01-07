# Steering Angle Prediction for Self-Driving Cars

A pytorch implementation of NVIDIA's 2016 paper "End to End Learning for Self-Driving Cars". 

---

### Test Log

#### Multi Frame Model (01/06/2021)

I would consider this model my first major improvement over Model 1. It reaches validation set losses in the 200s within 10 epochs (as opposed to a minimum of 410.72 reached by Model 1 after 130 epochs). Training loss lags behind a bit, but appears to be declining at the end of training as the model overfits this the training set (as it does in every model so far). So... what is this model?

This version of the "Multi Frame Model" is simply "Deeper Model", but it takes the previous 4 frames as input along with the current frame, giving it some temporal information. I implemented this by simply concatenating each frame along the rgb dimension, so while Deeper Model takes input tensors (frames) of size [batch_size, 3, 455, 256], this model takes tensors of size [batch_size, 15, 455, 256]. This led to surprisingly good results, as it beat all previous models on the validation set by a significant margin.

<img src="https://github.com/whwiese/SteeringAnglePrediction/blob/master/ModelStats/MultiFrame/Loss35e.png" alt="gen" width="400"/> <img src="https://github.com/whwiese/SteeringAnglePrediction/blob/master/ModelStats/MultiFrame/ValSet35.png" alt="gen" width="400"/>

We can see that the model performs well on the validation set relative to the other models, but it still has an oversteering problem on large turns. Given the small amount of data I'm training on, however, I'd say this is a strong result. Downside is it takes a long time to train due to slow data loading. One could imagine a more efficient data collection and preprocessing procedure in a real car though. Check out the MultiFrame folder in ModelStats for more plots and info.

What's next? Probably some messing with normalizers (is batch norm helping? add dropout?), and hyperparameters (learning rate, weight decay), then I'd like to explore LSTM models.

Note: I realize 3D convolutions are a thing... and are desdigned specifically for datasets like this. I will likely experiment with using these in the future too.

#### Deeper Model (01/04/2021)

I added some extra conv layers to model 1 and trained this as a baseline for futute, more ambitious, architectures. Training went slower than with model 1, but this model eventually reached the same loss levels and even a bit lower, though I let it train for longer. See the DeeperModel folder in ModelStats for plots and info.

#### Diff Model (01/02/2021)

One issue with Model 1 is that it lacked any temporal information in its training process. It does not treat a frame differently based on information from previous frames. For this model I introduced some temporal information by inputting the pixel-by-pixel differnece between the current frame and the next-to-previous frame (inspired by [ref 3](https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/rambo)), as opposed to simply inputting the current frame. I chose to look back two frames instead of just one so the difference between frames would be more significant (the data used is about 20 fps), although I have included "lookback" as a hyperparameter of the custom dataset so I can adjust it in future tests. 

During training, the losses on the training and validation sets initially decreased more rapidly than they did when trainign Model 1, but training appeared to saturate around 60 epochs, and while the model reached similar training set loss to Model 1, it never quite reached Model 1's lowest validation set loss mark. For training and performance graphs check out the DiffModel folder in the ModelStats folder in this repository.

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

### Guide To Files

model.py: Contains pytorch model definitions.

modelUtils.py: Utility functions for efficient model building. Only used within model.py.

dataset.py: Contains custom dataset classes.

train.py: An example of the loop used to train models. Can make use of any datasets/models in this repository with slight modifications.

utils.py: Utility functions used by other files in the repository.

---

### References

1. [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) - Bojarski et al., 2016.
2. [Driving Data](https://github.com/SullyChen/driving-datasets) - Sully Chen.
3. [rambo Udacity Challenge Model](https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/rambo)
