# Unit test for function 'xldnew'
import numpy as np
from numpy import loadtxt
from xldnew import xldnew

if __name__ == "__ main__":

    NOAREA = 2
    PKLOAD = np.array([[6000, 9000]]).T
    FileNameAndPath = "tests/unit_test_input_for_xldnew"
    HRLOAD = xldnew(FileNameAndPath, PKLOAD)
    HRLOAD_truth = loadtxt("tests/ground_truth_for_xldnew.txt", unpack=False)

    np.testing.assert_array_equal(HRLOAD, HRLOAD_truth.T)
