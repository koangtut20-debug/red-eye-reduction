from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def reduce_red_eye(input_path, output_path):
    image = cv2.imread(input_path)

    if image is None:
        return False

    corrected = image.copy()

    # Region around the affected eye
    x1, y1 = 540, 210
    x2, y2 = 620, 275

    # Make sure the image is large enough
    height, width = image.shape[:2]

    if x2 > width or y2 > height:
        return False

    eye = image[y1:y2, x1:x2].copy()

    # Convert to HSV
    hsv = cv2.cvtColor(eye, cv2.COLOR_BGR2HSV)

    # Detect red pixels
    lower_red1 = np.array([0, 45, 70])
    upper_red1 = np.array([12, 255, 255])

    lower_red2 = np.array([168, 45, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    # Keep detection inside the eye
    oval_mask = np.zeros_like(red_mask)

    cv2.ellipse(
        oval_mask,
        (40, 32),
        (32, 20),
        0,
        0,
        360,
        255,
        -1
    )

    red_mask = cv2.bitwise_and(
        red_mask,
        oval_mask
    )

    # Remove tiny noise
    kernel = np.ones((2, 2), np.uint8)

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Smooth the mask
    soft_mask = cv2.GaussianBlur(
        red_mask,
        (7, 7),
        0
    )

    alpha = soft_mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]

    # Split image channels
    b, g, r = cv2.split(eye)

    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    # Find extra red
    red_excess = r - g

    reduction = np.clip(
        red_excess * 0.65,
        0,
        80
    )

    r_new = np.clip(
        r - reduction,
        0,
        255
    )

    corrected_eye = cv2.merge([
        b.astype(np.uint8),
        g.astype(np.uint8),
        r_new.astype(np.uint8)
    ])

    # Blend correction smoothly
    result_eye = (
        eye.astype(np.float32) * (1 - alpha)
        + corrected_eye.astype(np.float32) * alpha
    ).astype(np.uint8)

    # Put corrected eye back
    corrected[y1:y2, x1:x2] = result_eye

    # Save result
    cv2.imwrite(output_path, corrected)

    return True


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():

    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]

    if file.filename == "":
        return "No image selected", 400

    filename = secure_filename(file.filename)

    input_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "original_" + filename
    )

    output_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "corrected_" + filename
    )

    file.save(input_path)

    success = reduce_red_eye(
        input_path,
        output_path
    )

    if not success:
        return "Unable to process this image.", 400

    return render_template(
        "result.html",
        original="original_" + filename,
        corrected="corrected_" + filename
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


if __name__ == "__main__":
    app.run(debug=True)