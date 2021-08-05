#Handles arguments to the command terminal.

import argparse

    
def get_input_args_train():
    """
    Retrieves and parses the 8 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 6 command line arguments. If 
    the user fails to provide some or all of the 6 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      0. The image directory, data_dir is defined at the front.
      1. Directory to save as --save_dir with default value, save_directory (str)
      2. Save folder directory as --folder with default ''
      3. CNN Model Architecture as --arch with default value 'densenet161', but could be 'vgg13' etc.
      4. Learning rate as --learning_rate with default 0.001, but could be 0.01 etc.
      5. Input to hidden layer as --hidden_units with default 512
      6. Number of epochs as --epochs and default 20
      7. GPU available as --gpu default False
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    #00
    parser.add_argument('data_dir', help = 'data directory')
    
    #01
    parser.add_argument('--save_dir', type = str, default = '/home/workspace/ImageClassifier/trained_flower_models/', 
                    help = 'path to current working folder')
    
    #02
    parser.add_argument('--save_folder', type = str, default = '', 
                    help = 'path to folder from save_dir, forward slash at end. E.g. complete path = save_dir + [save_folder/]')
    
    #03
    parser.add_argument('--arch', type = str, default = 'densenet161', 
                    help = 'model architecture', choices=['vgg13', 'vgg19', 'densenet121', 'densenet161'])
    #offer choices later
    #parser.add_argument("--arch", type=str, default="vgg", help="cnn architecture to use", choices=['vgg', 'alexnet', 'resnet'] )
    
    #04
    parser.add_argument('--learning_rate', type = float, default = 0.003, 
                    help = 'learning rate for optimizer')
    
    #05
    parser.add_argument('--hidden_units', type = int, default = 512, 
                    help = 'number of inputs to hidden layer')
    
    #06
    parser.add_argument('--epochs', type = int, default = 10, 
                    help = 'number of epochs for training')
    
    #07
    parser.add_argument('--gpu', action = "store_true", default = False, 
                    help = 'presence of gpu for computation')
    
    # This prints to command, for testing.
    args = parser.parse_args()
    print('data_dir value:', args.data_dir)
    print('Gpu value:', args.gpu)
    
    return parser.parse_args()