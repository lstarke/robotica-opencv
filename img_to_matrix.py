import cv2
import numpy as np

original_img_path = 'C:\\Users\\Starke\\Desktop\\cenario.jpg'
kernel = np.ones((5, 5), np.uint8)
matrix = np.zeros((7, 7))

img_in_grayscale = cv2.imread(original_img_path, 0)
ret_val, img_with_thresh_binary = cv2.threshold(img_in_grayscale, 110, 255, cv2.THRESH_BINARY)
img_with_dilation = cv2.dilate(img_with_thresh_binary, kernel, iterations=1)
img_with_erosion = cv2.erode(img_with_dilation, kernel, iterations=1)


def scan_matrix_cell(img, row, column):
    img_rows, img_columns = img.shape
    max_rows = int((img_rows / 7) * (row + 1))
    max_columns = int((img_columns / 7) * (column + 1))
    min_row = int(max_rows - (img_rows / 7))
    min_column = int(max_columns - (img_columns / 7))
    black_pixels = 0
    while min_row < max_rows:
        if min_row == 47 and column == 1:
            print()
        while min_column < max_columns:
            if img_with_erosion[min_row, min_column] == 0:
                black_pixels = black_pixels + 1
            min_column = min_column + 1
        min_row = min_row + 1
        min_column = int(max_columns - (img_columns / 7))
    if ((black_pixels * 100) / (max_rows * max_columns)) > 0.34:
        return 1
    return 0


for row in range(7):
    for column in range(7):
        matrix[row, column] = scan_matrix_cell(img_with_erosion, row, column)

print(matrix)
np.savetxt("matrix.csv", matrix, delimiter=",")

# uncomment to show the images with applied filters

# cv2.imshow('img_in_grayscale', img_in_grayscale)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('img_with_thresh_binary', img_with_thresh_binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('img_with_dilation', img_with_dilation)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('img_with_erosion', img_with_erosion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

