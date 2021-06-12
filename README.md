# Bird Classifier
Bird classifier for a [Kaggle competition](https://www.kaggle.com/c/birds21sp/) as a final project for our CSE 455: Computer Vision class.
<br />
A video summary can be found [here](#).

## Introduction
* What are birds? Which bird is it? The goal of this challenge is to classify images of birds. The Kaggle data set we used has a total of 555 different image categories (birds), and the final test set has 10,000 images.

## Process
* Algorithms/Techniques:
  * Convolutional Neural Networks
  * Transfer Learning
  * Data Augmentation
* Used PyTorch for creating our models, mostly following the tutorials and modifying a pretrained ResNet model.
* Tried to use local GPU (since our GPUs are quite good) to see if we could run things faster, but ran into issues so we ended up sticking with Google Colab's free GPU.
  * Training our network was very slow...  mistakes and small adjustments were very time consuming
  * Ran into some issues with Colab like CUDA running out of memory
* Tested mostly independently on our own Google accounts to save computation time while sharing some changes that seemed to work (not the most "scientific" but we were on a bit of a time crunch)
  * Ideally should have tested changes one at a time, but usually ended up not because we were a bit limited on time
* Tested with both ResNet-18 and ResNet-50 
  * Upon some research, we learned ResNet-50 has better accuracy without the worst computation time trade-off
    * Had to reduce batch size to fit on Colab GPU
  * Did a bit more experimenting with ResNet-18 as it was faster
    * Got highest of 0.73900 on Kaggle
  * Increased image size from 128 to 256 - higher resolution should help pick out details
  * Normalized input images according to [documentation](https://pytorch.org/vision/stable/models.html)
    * Normalized input seemed to to work fine with ResNet-18, but not with ResNet-50 (or we made some sort of mistake along with it, but removing that part seemed to make it significantly better - went from 32% to 81% on Kaggle???)
  * Adjusted epochs (with a lower learning rate later)
    * More epochs gives more time for the network to learn - but not always better
    * Preferably change learning rate when the loss plateaus
* Didn't make a validation set at first - submitting to Kaggle somewhat blindly 
  * Did figure out a way to do validation testing, but didn't end up being used in our final submitted version as we didn't want to re-train everything on ResNet-50 (and we were satisfied enough with our Kaggle results)
  * Used Linux commands to select 3 random files from each numbered subfolder (type of bird) in the training folder, moving them to a corresponding subfolder in the validation folder, and then retraining (so none of the validation images would get mixed up with the training set)


## Results
* Highest score of 0.81400 on Kaggle
  * One of our ResNet-50 versions
    * Resnet-50 version that used 7 epochs performed better than version using 12 (0.81400 vs 0.77000 on Kaggle); 7 epochs also resulted in lower final loss - around 0.3 vs around 0.5  
  * Had higher training accuracy (94%), but worse test accuracy on Kaggle
    * Model likely overfitted to training set
* Probably could have tuned our parameters a bit more with testing to fix overfitting, but it was kind of painfully slow (especially with ResNet-50) and we had limited time
  * Could have tried to add more data augmentation to reduce overfitting (e.g. RandomResizedCrop or RandomErasing)
* Graphs of loss can be seen in ipynb files on GitHub

## Conclusion
* We learned how to use Colab and PyTorch to train a bird classifier
  * Learned about pre-trained models (so you don't have to do it from scratch)
  * Learned how to do validation and why it might be useful (though we didn't end up utilizing it)
  * Training neural networks takes a long time :(
* This project could be extended to really any type of classifier in general (e.g. dogs, cats, etc.)
