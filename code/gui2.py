import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

model = load_model('Traffic sign classification\my_model.h5')

classes = {    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',
    5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)',
    7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',
    9:'No passing',
    10:'No passing veh over 3.5 tons',
    11:'Right-of-way at intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No vehicles',
    16:'Veh > 3.5 tons prohibited',
    17:'No entry',
    18:'General caution',
    19:'Dangerous curve left',
    20:'Dangerous curve right',
    21:'Double curve',
    22:'Bumpy road',
    23:'Slippery road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic signals',
    27:'Pedestrians',
    28:'Children crossing',
    29:'Bicycles crossing',
    30:'Beware of ice/snow',
    31:'Wild animals crossing',
    32:'End speed + passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep right',
    39:'Keep left',
    40:'Roundabout mandatory',
    41:'End of no passing',
    42:'End no passing veh > 3.5 tons'}

root = tk.Tk()
root.title("Traffic Sign Classifier")
root.geometry("850x650")
root.configure(bg="#f0f2f5")
root.minsize(700, 550)



style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12, "bold"), padding=6)
style.map("TButton",
          foreground=[('active', 'blue')],
          background=[('active', '#1f77b4')])

title_label = tk.Label(root, text="Know Your Traffic Sign",
                       font=("Helvetica", 26, "bold"), fg="#364156", bg="#f0f2f5")
title_label.pack(pady=20)

image_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove", padx=10, pady=10)
image_frame.pack(pady=15, expand=True)
sign_image = tk.Label(image_frame, bg="#ffffff")
sign_image.pack(expand=True)

result_label = tk.Label(root, text="", font=("Helvetica", 18, "bold"),
                        fg="#011638", bg="#f0f2f5")
result_label.pack(pady=15)

button_frame = tk.Frame(root, bg="#f0f2f5")
button_frame.pack(pady=20)

def classify(file_path):
    image = Image.open(file_path).resize((30, 30))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    pred_prob = model.predict(image)[0]
    pred_class = np.argmax(pred_prob)
    sign_name = classes[pred_class]
    probability = pred_prob[pred_class] * 100

    result_label.config(text=f"{sign_name} ({probability:.2f}%)")

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        uploaded = Image.open(file_path)
        uploaded.thumbnail((400, 400))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.config(image=im)
        sign_image.image = im

        classify_btn.config(state="normal")
        classify_btn.file_path = file_path 

upload_btn = ttk.Button(button_frame, text="Upload Image", command=upload_image)
upload_btn.grid(row=0, column=0, padx=15)

classify_btn = ttk.Button(button_frame, text="Classify Image",
                          command=lambda: classify(classify_btn.file_path), state="disabled")
classify_btn.grid(row=0, column=1, padx=15)

#hover
def on_enter(e):
    e.widget.config(cursor="hand2")

def on_leave(e):
    e.widget.config(cursor="")

upload_btn.bind("<Enter>", on_enter)
upload_btn.bind("<Leave>", on_leave)
classify_btn.bind("<Enter>", on_enter)
classify_btn.bind("<Leave>", on_leave)

root.mainloop()
