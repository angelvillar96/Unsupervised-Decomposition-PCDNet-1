"""
Here we put the methods to augment and randomize the dataset samples
"""


def normalize_img(img):
    """ Performing image normalization """
    norm_img = (img - 0.5) * 2
    return norm_img


def unnormalize_img(img):
    """ Undoing image normalization """
    unnorm_img = (img / 2) + 0.5
    return unnorm_img


def rgb2gray(rgb):
    """ Converting RGB image to grayscale """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
