# Assignment 7

Train ResNet18 on Cifar10 for 20 Epochs

There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files your colab file must:

Train resnet18 for 20 epochs on the CIFAR10 dataset

Show loss curves for test and train datasets

Show a gallery of 10 misclassified images

Show gradcam Links to an external site.output on 10 misclassified images.

Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬 Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. Train for 20 epochs Get 10 misclassified images Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class) Apply these transforms while training: RandomCrop(32, padding=4) CutOut(16x16)

