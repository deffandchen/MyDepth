import numpy as np


filename = "utils/filenames/eigen_test_files.txt"
files = np.genfromtxt(filename, dtype=str, delimiter=' ')
print(filename)