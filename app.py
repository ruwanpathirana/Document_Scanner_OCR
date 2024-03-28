from flask import Flask, request, jsonify
import cv2
import numpy as np
from imutils.perspective import four_point_transform

app = Flask(__name__)

def resizer(image, width=500):
    h, w, c = image.shape
    aspect_ratio = w / h
    new_height = int(width / aspect_ratio)
    resized_image = cv2.resize(image, (width, new_height))
    return resized_image

def document_scanner(image):
    img_re = resizer(image)
    detail = cv2.detailEnhance(img_re, sigma_s=20, sigma_r=0.15)
    gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_image = cv2.Canny(blur, 75, 200)
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(edge_image, kernel, iterations=1)
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            four_points = np.squeeze(approx)
            break

    multiplier = image.shape[1] / img_re.shape[1]
    four_points_orig = four_points * multiplier
    four_points_orig = four_points_orig.astype(int)
    wrap_image = four_point_transform(image, four_points_orig)
    
    return wrap_image

@app.route('/', methods=['POST'])
def index():
    return "Welcome to Document Scanner!"

@app.route('/scan', methods=['POST'])
def scan_document():
    if request.method == 'POST':
        print("Request files:", request.files)

        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            try:
                img_np = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                print("Image shape:", img.shape)
                wrpimg = document_scanner(img)

                # Save the scanned document
                output_path = "./output/scanned_document.jpg"
                cv2.imwrite(output_path, wrpimg)
                print("Scanned document saved at:", output_path)

                return jsonify({'message': 'Document scanned successfully', 'output_path': output_path})
            except Exception as e:
                return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port=5000)
