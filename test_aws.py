import os
import cv2
from io import BytesIO
from PIL import Image
import csv
from google.cloud import storage
import boto3
import json
from tqdm import tqdm
def load_bucket_credentials(credential_file):
    with open(credential_file) as f:
        credentials = json.load(f)
    return credentials

def draw_grid_cloud_bucket(bucket_name, input_folder, output_folder, csv_file_path, credentials_file):
    # Explicitly pass the credentials file path to the storage client
    client = storage.Client.from_service_account_json(credentials_file)
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=input_folder)

    for blob in tqdm(blobs):
        if blob.name.endswith(('.jpg', '.jpeg', '.png')) and 'amazon' in blob.name.lower():
            local_path = os.path.join(output_folder, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            image = cv2.imread(local_path)
            red_boxes = detect_grid(image)
            processed_image_path = os.path.join(output_folder, f"processed_{os.path.basename(blob.name)}")
            cv2.imwrite(processed_image_path, image)
            os.remove(local_path)
            
            # Get the bucket link for the current image
            bucket_link = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"
            
            # Pass the bucket link to detect_text function
            detect_text(processed_image_path, csv_file_path, bucket_link)

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
    header_height = int(0.02 * height)  # Assuming header occupies 10% of the image height
    footer_height = int(0.1 * height)  # Assuming footer occupies 10% of the image height

    red_boxes = []

    # Iterate over contours
    for c in contours:
        # Calculate contour area
        area = cv2.contourArea(c)

        # Filter out contours based on area
        if area < 180:  # Adjust the minimum area threshold as needed
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

    print(f"Detected text from {image_path} saved to {csv_file_path}")

# Load bucket credentials
bucket_credentials_file = r"/Users/dhruvarora/Documents/Smollan : Google/onyx-oxygen-375915-b66e5ff41e06.json"
bucket_credentials = load_bucket_credentials(bucket_credentials_file)
bucket_credentials = {
    "bucket_name": "harmony_paas",
    "input_folder": "PaaS Data/digital_capture/27-04-24/Page Listings",
    "output_folder": r"/Users/dhruvarora/Documents/Smollan : Google/Listings Data Extraction"
}
csv_file_path = 'Version_1.csv'

with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Order', 'Box Coordinates', 'Detected Text', 'Bucket Link'])

draw_grid_cloud_bucket(
    bucket_credentials["bucket_name"],
    bucket_credentials["input_folder"],
    bucket_credentials["output_folder"],
    csv_file_path,
    r"/Users/dhruvarora/Documents/Smollan : Google/onyx-oxygen-375915-b66e5ff41e06.json"
)
