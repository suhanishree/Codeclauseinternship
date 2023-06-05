import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2

# Create the GUI window
window = tk.Tk()
window.title("Color Detection")
window.geometry("400x300")

# Function to open and process the image
def process_image():
    # Open a file dialog to choose the image file
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    # Read the image using OpenCV
    image = cv2.imread(file_path)
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Flatten the image into a 2D array
    pixels = image.reshape((-1, 3))
    
    # Calculate the average RGB values
    avg_red = int(pixels[:, 0].mean())
    avg_green = int(pixels[:, 1].mean())
    avg_blue = int(pixels[:, 2].mean())
    
    # Display the average RGB values
    messagebox.showinfo("Color Detection", f"Average RGB: ({avg_red}, {avg_green}, {avg_blue})")

# Button to open and process the image
process_button = tk.Button(window, text="Process Image", command=process_image)
process_button.pack(pady=10)

# Start the GUI main loop
window.mainloop()
