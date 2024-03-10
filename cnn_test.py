# import os
# import numpy as np
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.metrics import classification_report, confusion_matrix

# # Set the path to your test dataset
# test_dataset_path = os.path.join(os.path.dirname(__file__), 'test')

# # Load the trained model
# model = keras.models.load_model('_cnn_training_model_.h5')

# # Function to preprocess a single image
# def preprocess_image(img_path):
#     img = load_img(img_path, target_size=(256, 256))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0  # Normalize the pixel values
#     return img_array

# # Get the list of test image files
# test_image_files = [os.path.join(test_dataset_path, file) for file in os.listdir(test_dataset_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

# # Make predictions on each test image
# for img_file in test_image_files:
#     # Preprocess the image
#     img_array = preprocess_image(img_file)

#     # Make prediction
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions[0])
#     confidence = np.max(predictions[0]) * 100  # Confidence percentage

#     # Display the result
#     class_labels = list(train_generator.class_indices.keys())
#     predicted_class_label = class_labels[predicted_class]
#     print(f"Image: {os.path.basename(img_file)}, Predicted Class: {predicted_class_label}, Confidence: {confidence:.2f}%")




# import os
# import numpy as np
# from tensorflow import keras
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
# import json  # Import the JSON module

# # Set the path to your test images folder
# test_images_folder = os.path.join(os.path.dirname(__file__), 'test')

# # Load the trained model
# model = keras.models.load_model('_cnn_training_model_.h5')  # Provide the path to your saved model

# # Load the class labels from the training script
# with open('class_mapping.json', 'r') as json_file:
#     class_labels = json.load(json_file)

# # Function to preprocess a single image
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(256, 256))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array

# # Get the list of test image files
# test_image_files = [os.path.join(test_images_folder, file) for file in os.listdir(test_images_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

# # Make predictions on each test image
# for img_file in test_image_files:
#     # Preprocess the image
#     img_array = preprocess_image(img_file)

#     # Make prediction
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions[0])

#     # Print predicted class for debugging
#     print(f"Predicted Class Index: {predicted_class}")

#     # Get the class label
#     predicted_class_label = class_labels.get(str(predicted_class), 'Unknown')  # Use .get() to provide a default value if the key is not found

#     # Display the result
#     print(f"Image: {os.path.basename(img_file)}, Predicted Class: {predicted_class_label}")

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

# Set the path to your test images folder
test_images_folder = os.path.join(os.path.dirname(__file__), 'test')

# Load the trained model
model = keras.models.load_model('_cnn_training_model_.h5')  # Provide the path to your saved model

# Define the class labels from the training script
class_labels = ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot", 
                "Spider_mites_Two-spotted_spider_mite", "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", 
                "Tomato_mosaic_virus", "healthy", "powdery_mildew"]

# Function to preprocess a single image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Get the list of test image files
test_image_files = [os.path.join(test_images_folder, file) for file in os.listdir(test_images_folder) if file.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]

# Make predictions on each test image
for img_file in test_image_files:
    # Preprocess the image with data augmentation
    img = image.load_img(img_file, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Get the class label
    predicted_class_label = class_labels[predicted_class]

    # Display the result
    print(f"Image: {os.path.basename(img_file)}, Predicted Class: {predicted_class_label}")


