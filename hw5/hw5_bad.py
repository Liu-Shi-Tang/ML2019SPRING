import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image



# FGSM
def fgsmOnePicture(image, epsilon, data_grad):
    # Get element-wise sign of the data
    sign_data_grad = data_grad.sign()
    # FGSM
    acctackImg = image + epsilon * sign_data_grad
    return acctackImg


# test one picture
def testOnePicture (model, device, data, epsilon, lossFunc) :
    abortNum = 3
    oriData = data
    for j in range (abortNum) :
        # load data to device
        data = data.to(device)

        # Let data be changeable (Normally it is inchangeable in training stage)
        data.requires_grad = True

        # Get prediction
        output_prob_ori = model(data)
        output_label_ori = output_prob_ori.max(1, keepdim=True)[1]

        # Using loss function to calculate loss
        loss = lossFunc(output_prob_ori, output_label_ori[0])

        # reset grad of whole model
        model.zero_grad()

        # launch backward propagation
        loss.backward()

        # get gradients of input data (only meaningful while "data.requires_grad = True "")
        data_grad = data.grad.data

        # run FGSM
        acctackImg = fgsmOnePicture(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output_prob_attack = model(acctackImg)
        output_label_attack = output_prob_attack.max(1, keepdim=True)[1]

        # check if attack succeed
        if output_label_attack.item() != output_label_ori.item():
            return acctackImg, True
        else :
           data = attackImg.clone().detach()
            # data = torch.tensor(attackImg)  

    return oriData , False



# To determine the device I used
device = None
if torch.cuda.is_available() :
    device = torch.device("cuda")
else :
    device = torch.device("cpu")

# choosing resnet50 as my proxy model
model = resnet50(pretrained=True)
model.to(device)

# Using evaluate mode
model.eval()

# Using CrossEntropyLoss as loss function
lossFunction = nn.CrossEntropyLoss()

# attack picture
attack_pictures = []

# processing constant for resnet50
# reference : https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
processingFun = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

# To transform attack picture
revert_transform = transforms.Compose([
        transforms.Normalize((-mean / std), (1.0 / std))
    ])

# success count
successCount = 0

# parameter
epsilon = 0.005
picture_num = 200

# get input/output directories
inputDir = sys.argv[1]
if inputDir[-1] != '/' : 
    inputDir += '/'
outputDir = sys.argv[2]
if outputDir[-1] != '/' :
    outputDir += '/'

# attack each pictures
for i in range (picture_num) :
    img = Image.open(inputDir + '{:0>3d}.png'.format(i))
    img = processingFun(img).reshape((1, 3, 224, 224))
    attackImg, isSuccess = testOnePicture(model, device, img, epsilon, lossFunction)
    if isSuccess == True :
        successCount += 1
        print (str(i+1) + ' succeed !')
    else :
        print (str(i+1) + ' fail !')
    attack_pictures.append(attackImg)

# print success rate
print ("accuracy : " + str(successCount/picture_num))


# Save attack picture
for i in range(picture_num) :
    attack_img_PIL = revert_transform(attack_pictures[i].cpu().reshape((3, 224, 224)))
    # clamp to 0 ~ 1 
    attack_img_PIL = torch.clamp(attack_img_PIL, 0, 1)
    # Transform to image format (ps. other lib will output wrong result)
    attack_img_PIL = transforms.ToPILImage()(attack_img_PIL)
    # Save picture
    attack_img_PIL.save(outputDir + '{:0>3d}.png'.format(i))

