# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('your_image.jpg')

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Set a higher threshold to capture more dark pixels
# # Using a threshold value of 110
# _, black_mask = cv2.threshold(gray_image, 115, 255, cv2.THRESH_BINARY_INV)

# # Create a new image to keep only black pixels
# # Initialize the new image with white (or any other background color)
# result_image = np.full_like(image, 255)  # Create a white image

# # Use the black mask to keep only black pixels from the original image
# result_image[black_mask == 255] = image[black_mask == 255]

# # Save or display the result
# cv2.imwrite('only_black_pixels_higher_threshold_110.jpg', result_image)

# cv2.imshow('Only Black Pixels (Higher Threshold 110)', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2

# Load the image
# image = cv2.imread('your_image.jpg')

# # Convert to LAB color space
# lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# # Split the LAB channels
# L_channel, A_channel, B_channel = cv2.split(lab_image)

# # Increase the L channel (lightness)
# L_channel = cv2.convertScaleAbs(L_channel, alpha=1.5, beta=0)  # Increase lightness

# # Merge the channels back
# enhanced_lab_image = cv2.merge([L_channel, A_channel, B_channel])

# # Convert back to BGR color space
# enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

# # Save or display the enhanced image
# cv2.imwrite('enhanced_image.jpg', enhanced_image)
# cv2.imshow('Enhanced Image', enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# import cv2

# # Load the image
# image = cv2.imread('your_image.jpg')

# # Convert to LAB color space
# lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# # Split the LAB channels
# L_channel, A_channel, B_channel = cv2.split(lab_image)

# # Apply CLAHE to the L channel
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# L_channel_clahe = clahe.apply(L_channel)

# # Merge the channels back
# enhanced_lab_image = cv2.merge([L_channel_clahe, A_channel, B_channel])

# # Convert back to BGR color space
# enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

# # Save or display the enhanced image
# cv2.imwrite('clahe_enhanced_image.jpg', enhanced_image)
# cv2.imshow('CLAHE Enhanced Image', enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('your_image.jpg')

# # Define a sharpening kernel
# sharpening_kernel = np.array([[0, -1, 0],
#                                [-1, 5, -1],
#                                [0, -1, 0]])

# # Apply the sharpening filter
# sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

# # Save or display the sharpened image
# cv2.imwrite('sharpened_image.jpg', sharpened_image)
# cv2.imshow('Sharpened Image', sharpened_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('./1-1.jpg')

# # Increase contrast by 50%
# contrast_factor = 2.0
# contrast_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

# # Convert the contrast-enhanced image to HSV (Hue, Saturation, Value) color space
# hsv_image = cv2.cvtColor(contrast_image, cv2.COLOR_BGR2HSV)

# lower_red1 = np.array([0, 50, 50])
# upper_red1 = np.array([10, 255, 255])

# lower_red2 = np.array([170, 50, 50])
# upper_red2 = np.array([180, 255, 255])

# # Create masks for red areas
# mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
# mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
# red_mask = mask1 + mask2

# # Invert the mask to keep non-red areas
# non_red_mask = cv2.bitwise_not(red_mask)

# # Replace red areas with white
# white_image = np.full_like(image, 255)  # Create a white image
# result = cv2.bitwise_and(image, image, mask=non_red_mask)  # Keep non-red parts of the original image
# result_white = cv2.bitwise_and(white_image, white_image, mask=red_mask)  # Fill red parts with white
# final_result = cv2.add(result, result_white)

# # Save or display the final image
# cv2.imwrite('./contplusrep.jpg', final_result)
# cv2.imshow('Result', final_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('contrast_enhanced_image.jpg', contrast_image)
# cv2.imshow('Contrast Enhanced Image', contrast_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
from PIL import Image

import pytesseract
a  = pytesseract.image_to_string(Image.open('./only_black_pixels_higher_threshold_110.jpg'), lang='chi_tra')
print(a)
# print(type(a))
# print(pytesseract.get_languages(config=''))

