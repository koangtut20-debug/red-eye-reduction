import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("istockphoto-1325286533-1024x1024.jpg")

if image is None:
    print("Image not found")
else:
    corrected = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_eye.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray, 1.1, 5, minSize=(100, 100)
    )

    correction_mask = np.zeros(gray.shape, dtype=np.uint8)

    for (x, y, w, h) in faces:

        face_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            face_gray, 1.1, 5, minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes:

            eye_x = x + ex
            eye_y = y + ey

            cx1 = eye_x + int(ew * 0.20)
            cy1 = eye_y + int(eh * 0.20)
            cx2 = eye_x + int(ew * 0.80)
            cy2 = eye_y + int(eh * 0.80)

            eye_area = corrected[cy1:cy2, cx1:cx2]

            b, g, r = cv2.split(eye_area)

            red_mask = (
                (r > 120) &
                (r > g * 1.35) &
                (r > b * 1.35)
            )

            neutral = (
                (g.astype(np.float32) +
                 b.astype(np.float32)) / 2
            )

            new_r = (
                r.astype(np.float32) * 0.25 +
                neutral * 0.75
            )

            r[red_mask] = new_r[red_mask].astype(np.uint8)

            corrected[cy1:cy2, cx1:cx2] = cv2.merge(
                (b, g, r)
            )

            correction_mask[cy1:cy2, cx1:cx2][red_mask] = 255

    cv2.imwrite(
        "red_eye_corrected.jpg",
        corrected
    )

    original_rgb = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )

    corrected_rgb = cv2.cvtColor(
        corrected, cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(correction_mask, cmap="gray")
    plt.title("Red-Eye Detection")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(corrected_rgb)
    plt.title("Corrected Image")
    plt.axis("off")

    plt.show()