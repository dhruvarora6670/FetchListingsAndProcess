import os
import cv2
from io import BytesIO
from PIL import Image
import csv
import json
from google.cloud import storage
import boto3
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI endpoint is running"}

@app.post("/load_bucket_credentials/")
async def load_bucket_credentials(credential_file: UploadFile):
    credentials = json.load(await credential_file.read())
    return JSONResponse(content=credentials)

@app.post("/draw_grid/")
async def draw_grid(bucket_name: str, input_folder: str, output_folder: str, credentials_file: UploadFile):
    # Store the credentials file temporarily
    credential_path = os.path.join("/tmp", credentials_file.filename)
    with open(credential_path, "wb") as f:
        f.write(await credentials_file.read())

    # Initialize Google Cloud Storage client
    client = storage.Client.from_service_account_json(credential_path)
    bucket = client.bucket(bucket_name)

    # Iterate through blobs and process the images
    blobs = bucket.list_blobs(prefix=input_folder)

    # Store results
    csv_file_path = os.path.join("/tmp", "Version_1.csv")
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'Order', 'Box Coordinates', 'Detected Text', 'Bucket Link'])

        for blob in blobs:
            if blob.name.endswith(('.jpg', '.jpeg', '.png')) and 'amazon' in blob.name.lower():
                local_path = os.path.join(output_folder, os.path.basename(blob.name))
                blob.download_to_filename(local_path)

                # Process the image
                image = cv2.imread(local_path)
                red_boxes = detect_grid(image)

                # Save the processed image
                processed_image_path = os.path.join(output_folder, f"processed_{os.path.basename(blob.name)}")
                cv2.imwrite(processed_image_path, image)

                # Remove the local file after processing
                os.remove(local_path)

                # Get the bucket link for the current image
                bucket_link = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"

                # Detect text and update CSV
                detect_text(processed_image_path, csv_file_path, bucket_link)

    return {"message": "Processed images and generated CSV", "csv_path": csv_file_path}


def detect_grid(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define image dimensions
    height, width = image.shape[:2]

    # Define header and footer regions to exclude
    header_height = int(0.02 * height)
    footer_height = int(0.1 * height)

    red_boxes = []

    # Iterate over contours
    for c in contours:
        # Calculate contour area
        area = cv2.contourArea(c)

        # Filter out contours based on area
        if area < 180:
            continue

        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(c)

        # Exclude contours in the header and footer regions
        if y < header_height or y + h > height - footer_height:
            continue

        # Draw rectangles around the detected boxes
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Store the coordinates of red boxes
        red_boxes.append((x, y, w, h))

    return red_boxes


def detect_text(image_path, csv_file_path, bucket_link):
    rekognition_client = boto3.client('rekognition')
    image = cv2.imread(image_path)

    red_boxes = detect_grid(image)

    if not red_boxes:
        return

    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        for idx, box in enumerate(red_boxes, start=1):
            x, y, w, h = box
            cropped_img = image[y:y+h, x:x+w]

            # Convert OpenCV image to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

            # Write PIL image to BytesIO
            image_bytes_io = BytesIO()
            pil_image.save(image_bytes_io, format='JPEG')

            # Get the byte representation
            image_bytes = image_bytes_io.getvalue()

            # Rekognition for text detection
            response = rekognition_client.detect_text(
                Image={'Bytes': image_bytes}
            )

            all_text = ""
            for text_detection in response['TextDetections']:
                all_text += text_detection['DetectedText'].strip() + " "

            writer.writerow([os.path.basename(image_path), idx, (x, y, w, h), all_text.strip(), bucket_link])
