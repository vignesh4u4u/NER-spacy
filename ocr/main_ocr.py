import os
import re
import json
import shutil
import uuid
import io

from flask import Flask, request, jsonify, json
from paddleocr import PaddleOCR
import numpy as np
import cv2
import pypdfium2 as pdfium

class OcrMethod:
    ocr = None

    def __init__(self):
        if OcrMethod.ocr is None:
            OcrMethod.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def process_ocr_result(self, result, fields_input):
        detected_text_list = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                detected_text_list.append(line[1][0])
        text = ' '.join(detected_text_list)
        data = {}
        fields = json.loads(fields_input)
        if fields_input:
            for field in fields:
                key = field.get("key")
                pattern = field.get("pattern")
                repeatable = field.get("repeatable", True)
                table = field.get("table")
                if table == False:
                    if pattern:
                        matches = re.findall(pattern, text, flags=re.IGNORECASE)
                        if matches and table == False:
                            if repeatable and table == False:
                                data[key] = matches
                            else:
                                data[key] = matches[0]
                if table == True:
                    table_pattern = json.loads(fields_input)
                    matched_data = {}
                    for f in table_pattern:
                        key1 = f.get("key")
                        pattern1 = f.get("pattern")
                        repeatable1 = f.get("repeatable", True)
                        table1 = f.get("table")
                        if table1 == True:
                            if pattern1:
                                matches1 = re.findall(pattern1, text, flags=re.IGNORECASE)
                                if matches1 and table1 == True:
                                    matched_data[key1] = matches1
                    output_data = []
                    keys = list(matched_data.keys())
                    if keys:
                        max_entries = max(len(matched_data[key]) for key in keys)
                        for i in range(max_entries):
                            entry = {}
                            for key in keys:
                                if i < len(matched_data[key]):
                                    entry[key] = matched_data[key][i]
                            output_data.append(entry)
                        data['table_data'] = output_data
                    else:
                        data['table_data'] = "No matching data found for table patterns."

        return data

    def converterfile(self, file, fields_input):
        file_end = file.filename.endswith(".pdf")
        if file_end == False:
            image_data = file.read()
            fields = json.loads(fields_input)
            gray = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
            result = OcrMethod.ocr.ocr(image_data)
        else:
            file_path =io.BytesIO(file.read())
            pdf = pdfium.PdfDocument(file_path)
            n_pages = len(pdf)
            image_path_list = []
            for page_number in range(n_pages):
                page = pdf.get_page(page_number)
                pil_image = page.render(scale=3).to_pil()
                image_path = f"image_{page_number + 1}.png"
                pil_image.save(image_path)
                image_path_list.append(image_path)
            for page_number in range(n_pages):
                page = pdf.get_page(page_number)
                pil_image = page.render(scale=3).to_pil()
                image_path = f"image_{page_number + 1}.png"
                pil_image.save(image_path)
                image_path_list.append(image_path)
            detected_text_list = []
            for image_path in image_path_list:
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                result = OcrMethod.ocr.ocr(image)
            pdf.close()
            os.remove(image_path)
            print("Temporary image path library removed")
        data = self.process_ocr_result(result, fields_input)


        if data :
            return jsonify(data)
        else:
            return jsonify({"error": "No matching data found"})

