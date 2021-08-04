# PROGRAMMER: Jade Chantrell
# DATE CREATED:   3/8/2021                               
# REVISED DATE: 
# PURPOSE: Classifies flower images using a pretrained CNN model.

# Imports functions created for this program
from get_input_args_train import get_input_args_train

from transform_data import transform_data
#from predict import predict


# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import os

# For category mapping
import json

def main():
                
    # Define get_input_args function within the file get_input_args.py
    # This function retrieves 6 Command Line Arugments from user as input from
    # the user running the program from a terminal window. 
    in_arg = get_input_args_train()
    
    #data_dir = 'flowers' #hard-coded in here for testing argparse.
    
    trainloader, validloader, testloader = transform_data(in_arg.data_dir)
      
    # Build a pre-trained network
    model_arch = in_arg.arch
          
    if model_arch == 'vgg13' :
        model = models.vgg13(pretrained=True) 
        input_fts = 25088
        
    elif model_arch == 'vgg19' :  
        model = models.vgg19(pretrained=True)
        input_fts =  25088
        
    elif model_arch == 'densenet121' :
        model = models.densenet121(pretrained=True) 
        input_fts = 1024
        
    elif model_arch == 'densenet161' :
        model = models.densenet161(pretrained=True) 
        input_fts = 2208
        
    # Block out the default, not needed.
    #else :
    #model = models.densenet161(pretrained=True) 
      
    # Print for checking
    #print(model)
      
    # Freeze parameters so as not to backpropogate through them
    for param in model.parameters():
        param.requires_grad = False
     

    # TEST PARAMETERS
    # output units are hard-coded (specific to the directory)
    drp = 0.2 #from 0.5
    hidden_unit = in_arg.hidden_units
    epochs = in_arg.epochs
    lr = in_arg.learning_rate
    save_dir = in_arg.save_dir
    save_folder = in_arg.save_folder
    gpu = in_arg.gpu
        
    
    #if not os.path.isfile('./checkpoint.pth'):
        
    print('Entering Build, Train, Verfiy, Test, Save model specified')
    print('Number of epochs is {}'.format(epochs))
        
    model.classifier = nn.Sequential(nn.Linear(input_fts, hidden_unit),
                                     nn.ReLU(),
                                     nn.Dropout(drp),
                                     nn.Linear(hidden_unit, 102),
                                     nn.LogSoftmax(dim=1))   

    # -------------------------------------------------------------------------
    # Allow model to be model to GPU if available
    if gpu :
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
    else :
        device = "cpu"
        model.to(device)
        
    #--------------------------------------------------------------------------  
        
    # Error criterion
    criterion = nn.NLLLoss()

    # Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr)
        
    # TRAIN
    from workspace_utils import active_session
        
    with active_session():
        
        #epochs defined above
        steps = 0
        running_loss = 0
        print_every = 10 #5 #64/4 so 4 per epoch.
        
        for epoch in range(epochs):
            
            # TRAINING
            for inputs, labels in trainloader:
                steps += 1

                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                #VALIDATION (TESTING)
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0

                    #Turns off dropout
                    model.eval()

                    with torch.no_grad():

                        for inputs, labels in validloader:

                            # Move input and label tensors to the default device
                            inputs, labels = inputs.to(device), labels.to(device)

                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(validloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(validloader):.3f}")

                    #resets the running loss for the next step loop
                    running_loss = 0

                    #turns back on dropout, and gradients I'm assuming.
                    model.train()
                            
    # TESTING NETWORK
    print('\n -------------------------\n')
    print('       Enter Testing')
    print('\n -------------------------\n')

    from workspace_utils import active_session

    with active_session():
        
        test_loss = 0
        accuracy = 0

        # Turns off dropout
        model.eval()

        with torch.no_grad():
            
            for inputs, labels in testloader:
                

                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # Printing for testing
        print(f"Test loss: {test_loss/len(testloader):.3f}.. "
              f"Test accuracy: {accuracy/len(testloader):.3f}")

        # Turns back on dropout, and gradients I'm assuming.
        model.train()
                        
    # Save checkpoint
                
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'index' : model.class_to_idx,
                  'model_type' : model_arch,
                  'epochs' : epochs,
                  'optimizer' : optimizer.state_dict,
                  'input_size' : input_fts,
                  'output_size' : 102,
                  'dropout_rate' : drp,
                  'learn_rate': lr,
                  'features' : model.features,
                  'classifier' : model.classifier,
                  'state_dict': model.state_dict()}
    
    # Save model

    torch.save(checkpoint, save_dir + save_folder + 'checkpoint.pth')

    print('--- Save complete ---')
                         
    #print(model)          
                
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
   

