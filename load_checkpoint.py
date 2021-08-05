def load_checkpoint(filepath):
    
    # Imports functions created for this program
    from get_input_args_train import get_input_args_train
    from get_input_args_predict import get_input_args_predict
                                      
    # Import necessary packages
    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
    
    # Define get_input_args function within the file get_input_args.py
    # This function retrieves 6 Command Line Arugments from user as input from
    # the user running the program from a terminal window. 
    in_arg = get_input_args_predict()
    gpu = in_arg.gpu
    model_arch_predict = in_arg.model_arch_predict
    
    # Allow model to be model to GPU if available.
    if gpu :

        map_location = "cuda:0" if torch.cuda.is_available() else "cpu"

    else :

        map_location = "cpu"
    
    #checkpoint = torch.load(filepath)
    #checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    #checkpoint = torch.load(checkpoint_file, map_location=('cuda' if (gpu and torch.cuda.is_available()) else 'cpu'))
    #checkpoint = torch.load(filepath, map_location=("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(filepath, map_location)
    
    #already loaded the correct model in train.py
    #model = models.densenet121(pretrained=True)
    
    # INITIALIZE LOAD, MODEL BUILD
    # Build a pre-trained network
      
    if model_arch_predict == 'vgg13' :
        model = models.vgg13(pretrained=True)
        
    elif model_arch_predict == 'vgg19' :
        model = models.vgg19(pretrained=True)

    elif model_arch_predict == 'densenet121' :
        model = models.densenet121(pretrained=True)

    elif model_arch_predict == 'densenet161' :
        model = models.densenet161(pretrained=True)
    
    model_epochs = checkpoint['epochs']
    model_input_size = checkpoint['input_size']
    model_output_size = checkpoint['output_size']
    model_type = checkpoint['model_type'] 
    model_drp = checkpoint['dropout_rate'] 
    model_lr = checkpoint['learn_rate'] 
    
    model.class_to_idx = checkpoint['index']    
    model.optimizer = checkpoint['optimizer']
       
    model.classifier = checkpoint['classifier']
    model.features = checkpoint['features']
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model