# Process a PIL image for use in a PyTorch model
                                                      
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    import numpy as np
    
    from PIL import Image    
    #with Image.open(image) as im:
    im = Image.open(image)
    
    #w = 256
    #h = 256
    #im = im.resize((w, h))
    
    (side1, side2) = im.size #instead of Image.size()
     
    #Set the size
    minsize = 256
    ratio = ( min(side1, side2) ) / minsize

    #size = int( (side1, side2)*(1/ratio) )
    side1 = int(side1*(1/ratio))
    side2 = int(side2*(1/ratio))
    size = (side1, side2)
    im.thumbnail(size)

    #cropping
    #(left, upper, right, lower)
    cropsize = 224

    #explicitly
    left = (side1 - cropsize) / 2
    upper = (side2 - cropsize) / 2
    right = ((side1 - cropsize) / 2) + cropsize
    lower = ((side2 - cropsize) / 2) + cropsize

    # Here the image "im" is cropped and assigned to new variable im_crop
    im_crop = im.crop( (left, upper, right, lower) ) #don't worry about int

    #converting the colour channels
    np_im = np.array(im_crop)/255
    
    #normalizing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_im = (np_im - mean) / std
    
    #reordering the colour channels - ToTensor already does this!
    #np_im = np_im.transpose(2, 0, 1)
                                              
    return np_im

# Displays a tensor image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax