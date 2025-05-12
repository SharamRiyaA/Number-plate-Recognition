Naan Mudhalvan: OpenCV-based Number Plate Recognition
This project demonstrates the use of OpenCV and Tesseract OCR to recognize and extract text from vehicle number plates. The project utilizes computer vision techniques for edge detection, contour identification, and text recognition in images uploaded to Google Colab.

Prerequisites
Before running the code, make sure to install the following dependencies:

Tesseract OCR for text recognition

OpenCV for image processing

Pytesseract for integrating Tesseract with Python

You can install the dependencies by running the following commands in your Google Colab environment:

bash
Copy
Edit
!apt-get install -y tesseract-ocr
!pip install pytesseract opencv-python
Overview
This project involves the following steps:

Image Upload: Users upload an image containing a vehicle number plate.

Preprocessing: The image is preprocessed to enhance features like edges using filters and thresholding.

Plate Detection: The program identifies the number plate by detecting contours and locating a rectangular shape within the image.

Text Recognition: After detecting the plate, the program uses Tesseract OCR to extract the text from the number plate.

Steps to Run the Code
Step 1: Install Dependencies
Install the necessary libraries:

bash
Copy
Edit
!apt-get install -y tesseract-ocr
!pip install pytesseract opencv-python
Step 2: Import Libraries
The required Python libraries are imported:

python
Copy
Edit
import cv2
import pytesseract
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow
Step 3: OCR Setup
Set up the path for the Tesseract executable:

python
Copy
Edit
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
Step 4: Upload Image
Use the Google Colab file upload tool to upload an image containing a vehicle number plate:

python
Copy
Edit
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
Step 5: Define Functions
Define various functions for image preprocessing, plate detection, and text recognition:

Preprocess Image: Converts the image to grayscale, applies bilateral filtering, and performs edge detection.

Find Plate Contour: Detects contours in the image and identifies the contour with four sides (which corresponds to the number plate).

Extract Plate: Extracts the portion of the image containing the number plate.

Recognize Plate Text: Applies OCR using Tesseract to extract text from the number plate.

Step 6: Process Image and Display Results
Load and process the image, detect the number plate, and display the results:

python
Copy
Edit
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
else:
    edged = preprocess_image(image)
    plate_contour = find_plate_contour(edged)

    if plate_contour is not None:
        plate_image = extract_plate(image, plate_contour)
        detected_text = recognize_plate_text(plate_image)

        print("Detected Number Plate Text:", detected_text)
        cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 3)
        cv2_imshow(image)
        cv2_imshow(plate_image)
    else:
        print("Number plate not detected.")
The program displays the processed image, highlighting the detected number plate, and also shows the extracted text.

Example Output
Upon successful execution, the system will output the recognized number plate text, such as:

yaml
Copy
Edit
Detected Number Plate Text: TN 22 AA 1234
The program also displays the original image with the detected number plate highlighted and the cropped portion showing the number plate.

Troubleshooting
No plate detected: If the number plate is not detected, check the quality and resolution of the image. High-resolution images with clear visibility of the plate yield better results.

Incorrect text extraction: The OCR model may have difficulty with some fonts or plate designs. Try adjusting preprocessing steps like thresholding or filtering to improve accuracy.

Conclusion
This project demonstrates how to leverage OpenCV and Tesseract OCR for automatic number plate recognition (ANPR). By using these techniques, you can easily integrate vehicle number plate recognition into your own projects or applications.
