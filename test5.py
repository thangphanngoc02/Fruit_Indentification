import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, filedialog
from tensorflow import keras
from keras.preprocessing import image

# Load the pre-trained model
model = keras.models.load_model('./fruit.h5')

# Function to preprocess the input image
def preprocess_image(img_path, target_size=(150, 150)):
    img_array = cv2.imread(img_path)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_array = cv2.resize(img_array, target_size)
    img_array = img_array / 255
    return img_array.reshape(-1, target_size[0], target_size[1], 3)  # Ensure 3 channels

# Function to make predictions
def predict_fruit(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]
    return predicted_class_index, confidence

# Function to open file dialog
def open_file_dialog():
    img_path = filedialog.askopenfilename(
        initialdir="/",
        title="Select file",
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*"))
    )
    if img_path:
        predicted_class_index, confidence = predict_fruit(img_path)
        predicted_class = categories[predicted_class_index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        img_label.config(image=img)
        img_label.image = img
        result_label.config(text=f'Predicted Fruit: {predicted_class}\nConfidence: {confidence:.2f}')

# Create the main window
root = Tk()
root.title("Fruit Prediction App")
root.geometry("1000x600")

# Label for displaying the image
img_label = Label(root)
img_label.pack(pady=20)

# Button to open file dialog
button = Button(root, text="Open Image", command=open_file_dialog)
button.pack(pady=20)

# Label for displaying the predicted fruit
result_label = Label(root, text="Predicted Fruit: ")
result_label.pack(pady=20)

# List of labels used during training (make sure this list is in the same order as the classes used during training)
categories = ['apple', 'banana', 'avocado', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'stawberries', 'watermelon']

# Run the Tkinter event loop
root.mainloop()
