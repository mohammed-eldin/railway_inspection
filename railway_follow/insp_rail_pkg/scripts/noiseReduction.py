import cv2
import os

# Path to the directory containing the input images
input_directory = 'resized_img/'

# Path to the directory where you want to save the processed images
output_directory = 'noise_reduction/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through each image in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image = cv2.imread(os.path.join(input_directory, filename))

        # Apply Gaussian blur for noise reduction
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Save the processed image to the output directory
        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, blurred_image)

        print(f"Processed and saved {filename}")

print("Processing and saving complete.")
