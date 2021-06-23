from sift import *
from surf import *
from orb import *

image = "Bancnote_redimensionate/500LEI/img1.jpg"
test_dir = r"Bancnote_redimensionate"
sift(image, test_dir)
surf(image, test_dir)
orb(image, test_dir)