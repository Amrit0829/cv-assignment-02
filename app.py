import cv2
import numpy as np
import streamlit as st

# Streamlit app
st.title("Image Segmentation App with OpenCV")
st.header("Upload an Image and Apply Segmentation Algorithms")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Segmentation options
st.subheader("Select Segmentation Algorithms")
thresholding = st.checkbox("Thresholding")
edge_detection = st.checkbox("Edge Detection")
color_segmentation = st.checkbox("Color Segmentation")
watershed_segmentation = st.checkbox("Watershed Segmentation")

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Error loading the image. Ensure it is a valid image file.")
    else:
        # Display the original image
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        # List to store processed images
        processed_images = []

        # Apply Thresholding
        if thresholding:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            processed_images.append(("Thresholding", thresh))

        # Apply Edge Detection
        if edge_detection:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            processed_images.append(("Edge Detection", edges))

        # Apply Color Segmentation
        if color_segmentation:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([35, 50, 50])  # Define range for segmentation
            upper_bound = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            segmented = cv2.bitwise_and(image, image, mask=mask)
            processed_images.append(("Color Segmentation", segmented))

        # Apply Watershed Segmentation
        if watershed_segmentation:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            markers = cv2.connectedComponents(sure_fg)[1]
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(image, markers)
            image[markers == -1] = [255, 0, 0]
            processed_images.append(("Watershed Segmentation", image))

        # Display processed images
        for name, img in processed_images:
            if len(img.shape) == 2:  # If grayscale, convert to RGB for display
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            st.image(img, caption=name, use_column_width=True)
