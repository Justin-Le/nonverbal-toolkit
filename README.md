# NVTK (Nonverbal Toolkit)

**Sample, fit, predict -- a toolkit for nonverbal expression recognition.**

## Motivation

Considering any configuration or movement of the hands, face, and head to be an "expression", the majority of techniques for expression recognition rely on fitting models to image/video data. What's wrong with this approach? Nothing, of course, assuming that the training data includes examples representative of the entire human race. It would be difficult to create such a dataset! A great number of people have features that appear nowhere in the currently available datasets, particularly people who have conditions such as:

* Cerebral Palsy
* Treacher Collins Syndrome
* Cleft Lip/Palate
* Severe Injury

and others that cause disfigurement. To make matters worse, even a single condition can have huge variation between examples, making it nearly impossible to create a dataset that represents, for instance, all individuals with cerebral palsy. For researchers and medical practitioners who aim to develop an expression recognition system that adapts to their individual patients, a software toolkit would greatly facilitate the process. Such a toolkit would need to encompass three major tasks: 

* collecting individualized data, tailored to the unique expressions of each patient
* fitting models to the data iteratively, to avoid having to refit on the entire dataset with every new addition of data
* making predictions on new expressions in real-time, to act as a means of instant communication between patient and caretaker

NVTK is a budding attempt to achieve this.

[See the video demo](https://youtu.be/Ttd7IJQdOhA), showing the result of these three steps on a very basic recognition task that has been solved numerous times in the literature. Demos on realistic cases will come soon.

The following documentation is intended for a rough prototype of the toolkit, where only facial expressions are considered. Future implementations will include recognition of hands, handheld items, and noises as means of expression, among other things.

## 1. Sample
**Track facial landmarks in real-time and store their coordinate trajectories.**

Usage:

`python sample.py [face_model]`

`face_model:` path to model trained on facial landmarks

The current default `face_model` is the default in [dlib](https://github.com/davisking/dlib), renamed as `models/face_predictor.dat`.

Each execution of `sample.py` generates a dataset for a new action class. Each action class is automatically stored in a separate CSV file under the data directory. Enter Ctrl-c at the terminal to end execution and generate these CSV files.

Examples of action classes: smile, frown, nod.

When answering the prompt:

* Enter 0 for the label on your first sampling session, 1 on the next, 2 on the next, and so on. In other words, the data directory should be populated with CSV files numbered from 0 upward once you've sampled all the action classes desired.
* Enter the same duration for each action, keeping in mind that an 8 indicates a duration of 8 frames, which lasts about 1 second.

## 2. Fit

**Fit a classifier to the data generated from sampling.**

Usage:

`python train.py [number of action classes] [classifier] [parameter1] [parameter2] [random_state]`

`classifier:` the method for classification, chosen from

* `lr`: logistic regression (default)
* `rf`: random forest

Options for parameters:

* `lr`: parameter1 is penalty (l1 or the default l2), parameter2 is regularization strength (positive float, default 1.0, with smaller values causing stronger regularization). 
* `rf`: parameter1 is number of estimators (default 10), parameter2 is max proportion of features used (positive float, default square root of number of features).

Note: `train.py` automatically outputs 5-fold cross-validation accuracies.

## 3. Predict
**Load fitted classifier and perform classification on real-time webcam footage.**

Usage:

`python predict.py [classifier]`

Choose `classifier` as `lr` for logistic regression or `rf` for random forest, depending on the one chosen when executing `train.py` previously.

Predictions are output to terminal.

## Requirements

* dlib
* scikit-learn
* opencv 2.4.13
* numpy
* pandas

It's recommended to install all of SciPy, which includes the last two requirements, as future modules may make use of other libraries in the stack.

## Laundry

* Simplify the interface for sampling such that action class labels are generated automatically and durations are remembered between sampling sessions
* Create an interface for tuning model parameters when running `train.py`
* Allow for loading previously saved model parameters when running train.py
* Create module for customizing cross-validation
* Check input of `train.py` for incorrect number of arguments
* Make all arguments optional for train.py
* Add description of real-time algorithm
