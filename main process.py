import os
from flask import Flask, request, jsonify, Response, make_response
from gevent.pywsgi import WSGIServer
from ocr import text_extract_pdf
from ocr import image_to_text
from ocr import  main_ocr
from ocr import main_ocr_coordinates
from ocr import main_ocr_drawbox
from ocr import main_code
CONTEXT_PATH = "/ml-service"
flask_app = Flask(__name__)
DEBUG = True


@flask_app.route(CONTEXT_PATH + "/health/v1/ping", methods=["GET"])
def health_check():
    return "pong"

@flask_app.route(CONTEXT_PATH + "/text-extraction/v1/extract", methods=["POST"])
def pdf_text_extraction():
    extractor = text_extract_pdf.PDFTextExtrator()
    print("inside the extraction")
    file = request.files["files"]
    selected_options = {}
    if ('extractOptions' in request.form):
        selected_options = request.form["extractOptions"]
    print (selected_options)
    response_data = extractor.extract_text(file, selected_options)
    return response_data

@flask_app.route(CONTEXT_PATH + "/image-to-text/v1/extract", methods=["POST"])
def image_text_conversion():
    converter = image_to_text.ImageToTextConverter()
    file = request.files["file"]
    return converter.convert(file)

@flask_app.route(CONTEXT_PATH + "/ocr/v1/extract", methods=["POST"])
def ocr_data_extraction():
    converter = main_ocr.OcrMethod()
    file = request.files["file"]
    fields_input = request.form["fields"]
    return converter.converterfile(file, fields_input)

@flask_app.route(CONTEXT_PATH + "/coordinates/v1/extract", methods=["POST"])
def image_text_coordinates():
    converter = main_ocr_coordinates.OcrMethodCoordinates()
    file = request.files["file"]
    return converter.coordinates(file)

@flask_app.route(CONTEXT_PATH + "/text-draw-box/v1/extract", methods=["POST"])
def image_text_boundingbox():
    converter = main_ocr_drawbox.OcrMethodDrawbox()
    file = request.files["file"]
    return converter.boundingbox(file)

if __name__ == "__main__":
    print("Starting the server on port 8080")
    flask_app.run(debug=True, host="0.0.0.0", port=8080)
    #http_server = WSGIServer(("", 8080), flask_app)
    #http_server.serve_forever()