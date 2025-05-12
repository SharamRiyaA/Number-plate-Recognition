## 🔍 Number Plate Detection Using OpenCV and Tesseract OCR
## 📝 Introduction
This project is focused on automatic number plate detection and recognition using OpenCV and Tesseract OCR. It was developed in Google Colab as part of the Naan Mudhalvan program. The goal is to detect a vehicle’s number plate from an uploaded image and extract readable text from it using image processing and optical character recognition (OCR) techniques.

## ⚙️ How the Project Works
The project follows these main steps:

## ✅ Step 1: Install Dependencies
python
Copy
Edit
!apt-get install -y tesseract-ocr
!pip install pytesseract opencv-python
✅ Step 2: Import Libraries
python
Copy
Edit
import cv2
import pytesseract
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow
## ✅ Step 3: OCR Setup
python
Copy
Edit
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
##  ✅ Step 4: Upload Image
python
Copy
Edit
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
## ✅ Step 5: Define Functions
python
Copy
Edit
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)
    return edged

def find_plate_contour(edged_image):
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4:
            return approx
    return None

def extract_plate(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(contour)
    plate_image = image[y:y+h, x:x+w]
    return plate_image

def recognize_plate_text(plate_image):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 8')
    return text.strip()
## ✅ Step 6: Load and Process Image
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
## 💻 Running the Project in Google Colab
You can run this entire project easily using Google Colab:

Open the notebook in Google Colab

Execute each cell in order

Upload an image of a vehicle when prompted

The image will be processed and the number plate text will be extracted and displayed

## 🧰 Technologies Used
Python

OpenCV – for image processing and contour detection

Tesseract OCR – for recognizing text from the number plate

Google Colab – cloud platform for easy code execution and testing

## 🟢 Sample Output
Detected Number Plate Text: TN09AB1234 (sample)

Displayed Images: Original image with number plate highlighted and cropped plate image

## 🏁 Conclusion
This project successfully demonstrates how we can detect and recognize vehicle number plates using Python, OpenCV, and Tesseract OCR. It's a practical application of computer vision that can be enhanced further for real-time systems such as traffic monitoring, parking access, and law enforcement.

Using Google Colab made the process easier, with no need for complex setup. This project was developed as part of the Naan Mudhalvan program, providing valuable hands-on experience in AI and image processing.

