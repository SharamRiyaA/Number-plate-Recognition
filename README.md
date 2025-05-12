
# 📸 Number Plate Detection using OpenCV and Tesseract

This project detects and reads number plates from images using OpenCV for image processing and Tesseract OCR for text recognition. It was developed as part of a Naan Mudhalvan project and executed in Google Colab.

---

## ✅ Introduction

Number plate recognition is an essential task in modern traffic systems, parking management, and surveillance. This project demonstrates how to detect and extract the text from number plates using Python libraries.

We use OpenCV for preprocessing the image and detecting contours, and Tesseract OCR for extracting the text from the detected number plate region.

---

## 🧰 Technologies Used

- Python
- OpenCV
- PyTesseract (Tesseract OCR)
- Google Colab

---

## ⚙️ Installation

```bash
!apt-get install -y tesseract-ocr
!pip install pytesseract opencv-python
```

---

## 📌 Steps Performed

1. **Install dependencies** in Google Colab.
2. **Upload an image** using the file upload interface.
3. **Preprocess the image** using grayscale conversion, bilateral filtering, and Canny edge detection.
4. **Detect contours** and find a possible rectangular contour corresponding to the number plate.
5. **Extract the plate region** using the contour and mask.
6. **Apply OCR** using Tesseract to extract the number plate text.
7. **Display results** using OpenCV's image display tools.

---

## ▶️ Running the Program in Google Colab

Just run each cell in order, and use the file upload cell to upload your test image. The code will automatically process the image and display the detected number plate and the recognized text.

---

## 📌 Output Example

- Detected number plate area highlighted in green.
- Printed extracted text from the plate.

---

## ✅ Conclusion

This project is a practical implementation of Computer Vision and OCR for real-world applications. It provides a foundation for more complex systems like automatic number plate recognition (ANPR) used in smart cities.

Using Google Colab simplifies the setup, allowing you to focus entirely on your code logic and testing.

---

## 📁 Project By

**Naan Mudhalvan Initiative Project**
