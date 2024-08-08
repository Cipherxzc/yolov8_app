from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

image_path = "C:\\Users\\tony_\\Downloads\\bus.jpg"

def main():    
    # Load a model
    model = YOLO("D:\\Cipherxzc\\Projects\\yolo\\yolov8_app\\models\\best.pt")  # build a new model from YAML

    # Predict the image
    results = model(image_path)

    # Get the processed image with bounding boxes
    processed_image = results[0]

    # Since we are processing a single image, we take the first element
    processed_image = processed_image[0]

    # Convert the image from BGR to RGB format for displaying with matplotlib
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(processed_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

if __name__ == '__main__':
    main()
