import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("istockphoto-1325286533-1024x1024.jpg")

if image is None:
    print("Image not found")
else:
    corrected = image.copy()

    # Region around the affected eye
    x1, y1 = 540, 210
    x2, y2 = 620, 275

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

    # Reduce only the red channel
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

    # Full-image detection mask
    detection = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    detection[y1:y2, x1:x2] = red_mask

    # Save corrected image
    cv2.imwrite(
        "red_eye_corrected.jpg",
        corrected
    )

    # Convert for display
    original_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    corrected_rgb = cv2.cvtColor(
        corrected,
        cv2.COLOR_BGR2RGB
    )

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(detection, cmap="gray")
    plt.title("Red-Eye Detection")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(corrected_rgb)
    plt.title("Corrected Image")
    plt.axis("off")

    plt.show()