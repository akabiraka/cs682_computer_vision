import cv2
import numpy as np
from problem_1_solver import Problem1Solver
from problem_2_solver import Problem2Solver
from problem_3_solver import Problem3Solver
from problem_4_solver import Problem4Solver
from problem_5_solver import Problem5Solver

directory = 'GaitImages/'
img1 = cv2.imread(directory + '00000048.png')
img2 = cv2.imread(directory + '00000173.png')


def get_binaryImg_and_boundaries(img):
    binary_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boundaries, _ = cv2.findContours(
        binary_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    boundaries = np.array(boundaries[0])
    return binary_img, boundaries


print("solving problem 1 ... ...")
p1 = Problem1Solver()
p1.solve(img1)
p1.solve(img2)

print("solving problem 2 ... ...")
p2 = Problem2Solver()
p2.solve()

print("solving problem 3 ... ...")
binary_img, boundaries = get_binaryImg_and_boundaries(img1)
p3 = Problem3Solver(binary_img=binary_img, boundaries=boundaries)
p3.solve()

print("solving problem 4 ... ...")
bin_img1, _ = get_binaryImg_and_boundaries(img1)
bin_img2, _ = get_binaryImg_and_boundaries(img2)
p4 = Problem4Solver()
p4.solve(bin_img1, '00000048.png')
p4.solve(bin_img2, '00000173.png')

print("solving problem 5 ... ...")
p5 = Problem5Solver()
p5.solve()

cv2.waitKey(0)
