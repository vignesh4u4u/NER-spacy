from flask import jsonify
import cv2
import numpy as np
from paddleocr import PaddleOCR


class ImageToTextConverter:
    ocr = None

    def __init__(self):
        if ImageToTextConverter.ocr is None:
            ImageToTextConverter.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def convert(self, file):
        print("Converting file: ", file)
        file_end = file.filename.endswith(".pdf")
        if file_end == True:
            return jsonify({"error": "PDF file detected, not supported yet"})
        else:
            image_data = file.read()
            gray = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2GRAY,
            )
            result = ImageToTextConverter.ocr.ocr(gray)
            detected_text_list = []

            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    detected_text_list.append(line[1][0])

            text = " ".join(detected_text_list)
            return jsonify({"text": text})
