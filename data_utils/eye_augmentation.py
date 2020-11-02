import cv2
import numpy as np


np.random.seed(0)


class CustomZoom(object):
    # Implements custom zoom  as described in the EyeKnowYou paper
    def __init__(self, kernel_size, min=0.1, max=2.0):

        # let's decide what kind of thing, (This section only should consist of paranmters)
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):

        #let's decide what to add here

        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample