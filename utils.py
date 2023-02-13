import torch
import numpy as np
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import cv2
import requests
from torchsummary import summary
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def plot_train_vs_test_accuracy(epochs, train_acc, test_acc):
  train_range = range(1,epochs+1)
  plt.plot(train_range, train_acc, 'g', label='Training accuracy')
  plt.plot(train_range, test_acc, 'b', label='validation accuracy')
  plt.title('Training and Validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()


def view_misclassified_images(model, device, dataset, classes):
  misclassified_images = []
  
  for images, labels in dataset:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
              if(len(misclassified_images)<20 and predicted[i]!=labels[i]):
                misclassified_images.append([images[i],predicted[i],labels[i]])
            if(len(misclassified_images)>20):
              break
    
  
  fig = plt.figure(figsize = (8,8))
  for i in range(20):
        sub = fig.add_subplot(5, 5, i+1)
        #imshow(misclassified_images[i][0].cpu())
        img = misclassified_images[i][0].cpu()
        img = img / 2 + 0.5 
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1, 2, 0)),interpolation='none')
        
        sub.set_title("P={}, A={}".format(str(classes[misclassified_images[i][1].data.cpu().numpy()]),str(classes[misclassified_images[i][2].data.cpu().numpy()])))
        
  plt.tight_layout()
  return misclassified_images
  

#https://jacobgil.github.io/pytorch-gradcam-book/CAM%20Metrics%20And%20Tuning%20Tutorial.html
#https://www.kaggle.com/code/antwerp/where-is-the-model-looking-for-gradcam-pytorch/notebook

def model_summary(model, input_size=(3, 32, 32)):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    model = model.to(device)
    summary(model, input_size=input_size)
	
def show_grad_cam(model, img, label, classes, mean, std):
    warnings.filterwarnings('ignore')
    prediction = model(img).data.cpu().numpy()
    prediction = np.argmax(prediction)
    print("Prediction is", classes[prediction])
    print("Label is", classes[label])
    img = img.data.permute(1, 2, 0).cpu().numpy()
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=mean, std=std)
    targets = [ClassifierOutputTarget(label)]
    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
    cam = np.uint8(255*grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])
    images = np.hstack((np.uint8(255*img), cam , cam_image))
    return Image.fromarray(images)

# Example usage:
# img = preprocess_image(np.float32(cv2.resize(img, (224, 224))) / 255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# print("Prediction is", classes[prediction])
# print("Label is", classes[label])
# visualize_grad_cam(model, img, label, classes)
