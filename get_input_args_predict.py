#Handles arguments to the command terminal.

import argparse

    
def get_input_args_predict():
    """
    Retrieves and parses the 6+ command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 6 command line arguments. If 
    the user fails to provide some or all of the 6 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      0. Name of the model used for training, to initiate the load.
      1. Point to input image as image_path with default value as /home/workspace/ImageClassifier/flowers/test_dir
      2. Point to image folder as --folder, with default at 10
      3. Checkpoint model with checkpoint variable, default is 'checkpoint.pth'
      4. Category names as --category_names, with deault cat_to_name.json
      5. The top_k values of the predictor as --top_k, with default as top_k = 5
      6. GPU available as --gpu default False
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    #00
    #default = '/home/workspace/ImageClassifier/flowers/test_dir'
    #TYPE into command line: flowers/test
    parser.add_argument('model_arch_predict', help = 'architecture used for training')
    
    #01
    #default = '/home/workspace/ImageClassifier/flowers/test_dir'
    #TYPE into command line: flowers/test
    parser.add_argument('image_dir', help = 'input image directory')
    
    #02
    parser.add_argument('--folder', type = str, default = '10', 
                    help = 'a string that is a number of the folder for image path')
    
    #03
    #default = '/home/workspace/ImageClassifier/trained_flower_models/checkpoint.pth'
    #TYPE into command line: 'trained_flower_models/checkpoint.pth'
    #TYPE into command line: checkpoint.pth
    parser.add_argument('checkpoint_path', help = 'path to saved checkpoint')
    
    #04
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'category mapping folder, get labels.')
    
    parser.add_argument('--top_k', type = int, default = 5, 
                    help = 'number of top_k probabilities')
    
    #05
    parser.add_argument('--gpu', action = "store_true", default = False, 
                    help = 'presence of gpu for computation')
    
    args = parser.parse_args()
    print('checkpoint_path value:', args.checkpoint_path)
    
    return parser.parse_args()