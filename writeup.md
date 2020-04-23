# **Traffic Sign Recognition** 

[//]: # (Image References)
[image1]: ./images/training_images.png "A sample of the training set images"
[image2]: ./images/training_histogram.png "A label frequency bar chart"
[image3]: ./images/extracted_images.png "Extracted images from Berlin and Bonn"

[image7]: ./extracted_signs/07.png "Correctly classified - 7"
[image10]: ./extracted_signs/10.png "Wrongly classified - 10"
[image15]: ./extracted_signs/15.png "Correctly classified - 15"
[image27]: ./extracted_signs/27.png "Correctly classified - 27"
[image29]: ./extracted_signs/29.png "Correctly classified - 29"

The objective of this project is to build a traffic sign classifier using TensorFlow.

### Data Set Summary & Exploration

The data, originally sourced from the [German traffic sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset), was separated into 34799 training images, 12630 test images and 4410 validation images.

The first thing we want to do, naturally, is get a look at some of the images we'll be dealing with. Here's a sample of 128, (one of the first things you'll see printed out in the [Jupyter notebook]()):

![A sample of the training set images][image1]

The 32 by 32 image resolution really stands out as awful, but that isn't the only thing that makes it hard to recognize these images. They seem rather dark. Confirming that observation, we converted the whole dataset to grayscale and took the average pixel value: 81.9, far less than the 128 we might expect. Examining R, G and B channel mean pixel values, we find 86.7, 79.5 and 81.8 - similar, but far less than 128. This will be a point to consider later when we want to regularize our data.

Next, we want to consider the labels and how well our dataset covers those labels. There are 43 distinct traffic sign labels, and there are at least 179 training images for every label. Notably, some traffic signs are an order of magnitude more common than the least common in our training data. However, that is not so unbalanced as to be problematic (indeed, this set is far more balanced than the real world, where the difference in prevalence between most and least common signs is closer to three orders of magnitude than to one).

![A label frequency bar chart][image2]


### Design and Test a Model Architecture

#### Preprocessing

In a preprocessing phaze, two possibilities were considered. The first was whether to convert all images to grayscale: that would reduce the number of parameters in the resulting model, and so reduced the risk of overfitting. On the other hand, color does provide additional information in road sign classification which I'm hesitant to throw away. My initial preference was to keep color, and results were good enough that I almost continued with that initial decision. I then switched to grayscale and saw no meaingful difference in the accuracy of predictions - and so have continued with grayscale (path of least resistance; it makes sense to focus on other areas!).

The second possibility - really an imperative for any neural network - is data normalization. As discovered above, the mean pixel values (across all channels) are far from 128. That deprives us of the default approach to image normalization: subtracting 128 from all pixel values, then dividing by 128, resulting in a value between -1 and 1. Instead, I computed the channel mean and standard deviation (for R, G and B channels respectively, across all images in the training set), then subtracted the channel mean divided by standard deviation across all pixels in the training set. When switching to grayscale, the same was done - except now on a single channel.

One further point should be mentioned here: the same preprocessing step will be required whenever the resulting model is used for classification - including for validation and testing. For this, we make use of the constant channel means and standard deviations that were calculated on the training set.

#### Model Architecture

We used TensorFlow to implement a neural network, following an architecture analogous to [LeNet](http://yann.lecun.com/exdb/lenet/).

Of course, this isn't 1998, so there were a few differences: we used more parameters (larger convolutional layers), which requires additional compute power and regulation to prevent overfitting. To regulate, we employed random dropout on each of the hidden layers.

The network followed an entirely sequential structure, as follows:
- two convolutional layers, each followed by relu activations and max-pooling. The first convolutional layer used a 2x2 kernal and 1x1 stride with valid padding, reducing each imgage from 32x32x1 to 28x28x32, then relu activation, then to 14x14x32 after max-pooling (2x2 kernel, 2x2 stride). The second layer reduced the image from 14x14x32 to 10x10x48 (again, with a 2x2 kernel and 1x1 stride), then relu activation, and then to 5x5x48 after max pooling (again, with a 2x2 kernel and 2x2 stride). The result was then flattened to a per-image vector of length 1200. Random dropout (with a keep probability of 0.6 while training, but 1.0 during classification) was applied to this vector.
- two fully connected hidden layers, each followed by relu activation and random dropout (with a keep probability of 0.6 while training, but 1.0 during classification). The first layer reduced each image from a vector of length 2000 to a vector of length 120; the second from 120 to 80.
- a final fully connected layer, reducing the per-image vector from length 80 to length 43. This corresponds to the one-hot vector used for representing image labels and image classification

For more detail, on this or on any other section, see the Jupyter notebook in this repository - [Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb).

Learning rate decay was considered, and would achieve faster convergence. However good results could be achieved by simply setting a low learning rate with a high number of training epochs, so this was the approach we took in practice.

Some reflection: this is very unlikely to be an optimal architecture for classification - it's likely that better results could be attained by making layers smaller but by going deeper with additional hidden layers. It's likely that better results could be attained with stacked convolutional layers with different kernels - analogous to the inception module approach. It would be fascinating to explore all of these options given more time, but this model proved sufficient for now.

#### Model Training

To train our model, we used an Adam Optimizer minimizing softmax cross-entropy loss.

Learning rate decay was considered, and would achieve faster convergence. However good results could be achieved by simply setting a low learning rate with a high number of training epochs, so this was the approach we took in practice: we applied a constant learning rate of 0.001; we used 800 epochs for training.

It was necessary to use a batched learning approach due to memory constraints and the size of the training set. I settled on a constant batch size of 8700 (approximately a quarter of the training set in each batch). Batches were shuffled between each epoch.

It's likely that better results could be attained by applying learning rate decay, or alternatively [batch size growth](https://research.google/pubs/pub46644/). And naturally beyond architecture: better results could be achieved with a larger training data set that more fully covers the variation of road signs in the real world. With more time, it would be fruitful to explore all of these points and their impact. For now, let's see what we can get with the architecture described.

#### Solution Approach

Before we move on to how this model performed, let's add some further clarification as to how it was developed. As stated, this model was strongly inspired with a LeNet architecture (classifying road signs is a very similar problem to classifying written characters, so where better to begin than LeNet?). Because thimages are a little more "complicated" than written characters (at least, they seem so), and because there are so many different types, I assumed we would benefit from more wider layers, especially in the convolutional layers. Thanks to TensorFlow and modern GPUs, adding those extra is straightforward, but is likely to require regularization (for the given data set). This reasoning was the basis for the model above, and there weren't many changes after that point.

An iterative approach was taken in setting the precise width of the convolutional layers, in setting the keep probability (in dropout) and in deciding whether to use color or grayscale preprocessed images as input to the model.

#### Results:

The final model results were:
* training set accuracy of 99.7%
* validation set accuracy 97.3%
* test set accuracy of 96.4%

That leaves some scope for improvement, but it's not bad at all. There may be a slight over-fit, but our model has not learned the training set entirely (which might be a sign of more sever over-fit), and accuracy on the validation and test sets are reasonably close to the training accuracy.

### Test a Model on New Images

What we really care about for a self-driving car, is the ability to reliably detect and classify real world road signs from uncontrolled driving environments. To approximate that, I found a couple of videos uploaded by motorcyclists driving through Berlin and Bonn (covering both of the post-WWII German capitals). Credit to the drivers for filming these:
[![Many traffic signs in Berlin](https://img.youtube.com/vi/Z1EdiuMJUJg.jpg)](https://www.youtube.com/watch?v=Z1EdiuMJUJg)
[![Many traffic signs in Bonn too](https://img.youtube.com/vi/KQfxd5hVJYY/0.jpg)](https://www.youtube.com/watch?v=KQfxd5hVJYY)

From that, I extracted frames including traffic signs, cropped those frames to a bounding box around the traffic signs and reduced the resolution to 32 by 32, then put together a file labelling these images. Here's what the extracted road signs from Berlin and Bonn looked like (I stopped at 30 images):
![extracted images from Berlin and Bonn][image3]

These seem to be in a reasonably similar format to the data set used for our model. There are however at least two noticable discrepencies: in some of these extracted images, the sign is at a substantial angle, not directly facing the camera. That differs from the training set samples I've examined. A sign at an odd angle (not facing the camera) might not trigger the appropriate activations in the convolutional layers of our model. The second apparent flaw applies to only one extracted image: the bottom left 70 km/h sign occupies less than half of the image, with plenty of tree and wall in the background to potentially confound our model.

Other reasons for caution include: these are images taken from different cameras, with uncorrected distortion and in diverse and different lighting conditions. Additionally the process of reducing these cropped frames to 32 by 32 pixels added some blur, and may differ from the sort of image used in our model.

#### New Images Results

First, the prediction accuracy for extracted images from Berlin and Bonn seems surprisingly good: 96.7%. That is impressive and comparable with the validation and test results (it's 3 decimal points better than the test set prediction accuracy!). Of course, this is a tiny non-representative sample from two cameras on short drives in two cities, and not statistically meaningful. It does however seem exciting: out of 30 random road sign images, 29 were correctly classified and one was not. Here's the breakdown:

| Image | Prediction | Prediction Confidence |
|:-----:|:----------:|:----------:|
| road works | road works | 100.0% |
| general caution | general caution | 100.0% |
| priority road | priority road | 100.0% |
| bumpy road | bumpy road | 100.0% |
| general caution | general caution | 100.0% |
| go straight or left | go straight or left | 99.5% |
| speed limit (50km/h) | speed limit (50km/h) | 97.9% |
| keep right | keep right | 100.0% | 100.0% |
| speed limit (30km/h) | speed limit (30km/h) | 100% |
| speed limit (50km/h) | keep left | 45.8% |
| go straight or right | go straight or right | 100.0% |
| priority road | priority road | 100.0% |
| keep right | keep right | 100.0% |
| road works | road works | 100.0% |
| bikes crossing | bikes crossing | 96.2% |
| speed limit (30km/h) | speed limit (30km/h) | 100.0% |
| yield | yield | 100.0% |
| road works | road works | 100.0% |
| ahead only | ahead only | 100.0% |
| turn right ahead | turn right ahead | 100.0% |
| go straight or right | go straight or right | 99.7% |
| go straight or right | go straight or right | 100.0% |
| yield | yield | 100.0% |
| priority road | priority road | 99.9% |
| no entry | no entry | 100.0% |
| no passing | no passing | 100.0% |
| speed limit (70km/h) | speed limit (70km/h) | 97.3% |
| priority road | priority road |  100.0% |
| speed limit (70km/h) | speed limit (70km/h) | 97.6% |
| speed limit (70km/h) | speed limit (70km/h) | 100.0% |
(Note: that "prediction confidence" is the highest softmax probability of our classifier..)

Squint, and you'll notice that the 50 km/h sign in the tenth row was wrongly classified as a "keep left" sign. To a human, that misclassification seems bizarre. Let's take a look at that sign:

![wrongly classified road sign][image10]

How would that possibly be construed as "keep left"? If we were to be generous, we might compare it with the other signs and notice that it is darker and with lower contrast than the others, it does appear badly blurred and a lot of the image is taken up by pixels that don't belong to the sign and may confuse things. And yet: one result like this destroys trust, especially without an understanding of how the wrong classification was arrived at. Perhaps the best aspect of that misclassification is that the confidence is just 45.8% - if we were to threshold this, and allow for a "can't classify" response, then our classifier might be more robust.

#### New Image Confidence - Alternative Contenders

Let's take another look at some of these images, and in the cases of lower confidence let's see the next most likely classifications from our model (ranked by declining softmax probability). Five of the above images had a prediction confidence/ probability of under 99%, so let's consider each of those in turn, in order of decreasing confidence.

Image 7 was correctly predicted as a speed limit (50km/h) sign, but with a confidence of just 97.9%. The rival classifications:

| Probability | Prediction |
|:---------------------:|:----:|
| 0.9790 | speed limit (50km/h) |
| 0.0110 | speed limit (80km/h) |
| 0.0048 | speed limit (30km/h) |
| 0.0041 | speed limit (60km/h) |
| 0.0002 | speed limit (100km/h) |

![classified road sign][image7]

That is interesting. It appears that our model has inferred the more general/ abstract properties of a speed limit sign. And the numbers "30" and "80" look more similar to "50" than other numbers do. Nice. 97.9% still feels like a robust confidence, and the other classifations up for consideration seem relatable (to a human).


Image 29 was correctly predicted as a speed limit (70km/h) sign, but with a confidence of just 97.6%. The rival classifications:

| Probability | Prediction |
|:---------------------:|:----:|
| 0.9763 | speed limit (70km/h) |
| 0.0201 | speed limit (30km/h) |
| 0.0013 | general caution |
| 0.0008 | keep left |
| 0.0005 | stop |

![classified road sign][image29]

Again, most of the classification weight that isn't taken by the correct classification, goes on another speed limit sign. Reasonable. This time however, general caution and keep left appear (perhaps because of the angle in the 7?). A stop sign also features. Let's hope we never mix "stop" with "drive at 70 km/h".


Image 27 was correctly predicted as a speed limit (70km/h) sign (again), but with a confidence of 97.3%. The rival classifications:

| Probability | Prediction |
|:---------------------:|:----:|
| 0.9731 | speed limit (70km/h) |
| 0.0191 | speed limit (30km/h) |
| 0.0030 | speed limit (120km/h) |
| 0.0008 | speed limit (50km/h) |
| 0.0005 | speed limit (100km/h) |

![classified road sign][image27]

This time, again, our model seems to be demonstrating a general "understanding" of speed limit signs. Neat.


Image 15 was correctly predicted as a bicycles crossing sign, but with a confidence of just 96.2%. The rival classifications:

| Probability | Prediction |
|:---------------------:|:----:|
| 0.9615| bicycles crossing |
| 0.0127 | slippery road |
| 0.0127 | children crossing |
| 0.0059 | bumpy road |
| 0.0020 | narrows on the right |

![classified road sign][image15]

These alternatives also allow a positive interpretation: our model seems to have a general idea of caution signs.


Finally, back to image 10, which should have been classified as a 50 km/h speed limit sign, but which was instead classified as a "keep left" sign with 45.8% probability. The rival classifications:

| Probability | Prediction |
|:---------------------:|:----:|
| 0.4577| keep left |
| 0.2367 | speed limit (30km/h) |
| 0.0719 | go straight or left |
| 0.0641 | general caution |
| 0.0533 | speed limit (70km/h) |

![wrongly classified road sign][image10]

From this list, the only label that feels right is "general caution". With probabilities so widely dispersed, it seems that this image does not strongly activate our model for any particular classification. It's not sufficiently clear why this is the case.

### Next Steps

It would be rewarding to explore how a wider variety of neural network architectures perform on this problem. It would be useful to read more literature too. And for evaluation purposes as well as for training, it would be nice apply computer vision methods to extract bounding-boxed road sign images from videos (these would still need to be manually labelled).
