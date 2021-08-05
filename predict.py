# PROGRAMMER: Jade Chantrell
# DATE CREATED:   3/8/2021                               
# REVISED DATE: 
# PURPOSE: Classifies flower images using a pretrained CNN model.

# Imports functions created for this program
from get_input_args_predict import get_input_args_predict
from load_checkpoint import load_checkpoint
from image_helper import process_image, imshow

# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import os
import random

# For category mapping
import json

def main():
    
    print('Make sure to run train.py first to train the model')
    
    # Define get_input_args function within the file get_input_args.py
    # This function retrieves 6 Command Line Arugments from user as input from
    # the user running the program from a terminal window. 
    in_arg = get_input_args_predict()
    
    gpu = in_arg.gpu
    top_k = in_arg.top_k
    category_names = in_arg.category_names
    checkpoint_path = in_arg.checkpoint_path
    
    # GET AN IMAGE
    folder = in_arg.folder
    image_dir = in_arg.image_dir
    image_path = image_dir + '/' + folder

    files=os.listdir(image_path)
    image=random.choice(files)

    image_path = image_dir + '/' + folder + '/' + image

    print('Selected image path:')
    print(image_path)
    
    # LOAD MODEL
    model = load_checkpoint(checkpoint_path)
    #print(model)
    print('--- Load complete ---')
    
    # Allow model to be model to GPU if available.
    if gpu :

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    else :

        device = "cpu"
        model.to(device)
        
    # PROCESS IMAGE
    input_image = process_image(image_path)
    
    # Converting to a tensor
    transform_ToTensor = transforms.Compose([transforms.ToTensor()])
    input_image = transform_ToTensor(input_image)
    
    # Has Torch Size [3, 224, 224] but we need it to be [1, 3, 224, 224] with 1 batch.
    input_image = input_image.unsqueeze(0)
    #input_image = input_image.to(dtype=torch.double)
    input_image = input_image.float()
    
    # Move image to selected device
    input_image = input_image.to(device)
    
        # FORWARD PASS
    model.eval()
    
    with torch.no_grad() :
        logps = model.forward(input_image)
    
    # Calculate probabilities
    ps = torch.exp(logps) #should be an array of 102 class probabilities by 1
    
    # This returns a tuple of the top- 𝑘  values and the top- 𝑘  indices
    top_p, top_class = ps.topk(top_k, dim=1) #sum along columns, but it's just one image here.
    
    #convert top_p and top_class to numpy arrays and flatten
    #need to be moved back to host memory first
    #top_p = top_p.to('cpu') - didn't work
    print(device)
    #if device == "cuda:0" :
    #    top_p = top_p.cpu()
    #    top_class = top_class.cpu()
    
    top_p = top_p.data.cpu().numpy().flatten()
    top_class = top_class.data.cpu().numpy().flatten()
    
    #top_p = top_p.numpy().flatten()
    #top_class = top_class.numpy().flatten()
    
    #create a dictionary
    #class_name1, idx  = model.class_to_idx.items()
    class_to_idx = model.class_to_idx #number to number, {"32": '0'}
    idx_to_class = { j : k for k,j in class_to_idx.items() }
    
    #array of strings
    class_numbers = []
    for i in range(len(top_class)) :
        class_numbers.append(idx_to_class[top_class[i]])
    
    print('These are the probs:')
    print(top_p)
    print('These are the classes:')
    print(class_numbers)
    
    # Getting the Class Labels                      
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
                                 
        #array of strings
        class_labels = []
        for i in range(len(class_numbers)) :
            class_labels.append(cat_to_name[class_numbers[i]])
        
    print('The flower name is: {}, with with probability of: {}'.format(class_labels[0], top_p[0]))
    
    return top_p, class_labels
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
   