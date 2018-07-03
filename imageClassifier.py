# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:08:39 2018

@author: Gautam Kumar
"""

import tkinter as tk
from tkinter import filedialog
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# load the model
model = VGG16()

window = tk.Tk()

window.title("VGG16 Image Classification ToolBox")
 
window.geometry('590x100')
 
lbl = tk.Label(window, text="Browse the Image") 
lbl.place(x=3, y=5)

txt = tk.Entry(window,width=50,state="normal")
txt.place(x=120, y=5)

message1 = tk.Label(window, text="") 
message1.place(x=3, y=35)
message2 = tk.Label(window, text="") 
message2.place(x=3, y=55)

def browse():
    fileName = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))    
    txt.insert(0,fileName)    
    
def imageClassification():
    imagePath = txt.get()
    
    try:

        txt['state'] = 'disabled'
        # load an image from file
        image = load_img(imagePath, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # predict the probability across all output classes
        yhat = model.predict(image)
        # convert the probabilities to class labels
        label = decode_predictions(yhat)
        # retrieve the most likely result, e.g. highest probability
        label = label[0][0]
        res1 = "Prediction (Probability)"
        res2 = str(label[1]) + " (" + str(round(label[2],2)) + ")"
        message1.configure(text = res1)
        message2.configure(text = res2)
    
        
    except:
        txt['state'] = 'normal'
        message1.configure(text = "Invalid Image Type or Path")
    

def clear():
    txt['state'] = 'normal'
    txt.delete(0, 'end')    
    res = ""
    message1.configure(text= res)
    message2.configure(text= res)
  
browseButton = tk.Button(window, text="Browse", command=browse)
browseButton.place(x=430, y=0)    
clearButton = tk.Button(window, text="Clear", command=clear)
clearButton.place(x=482, y=0)    
takeImg = tk.Button(window, text="Classify Images", command=imageClassification)
takeImg.place(x=430, y=35)
quitWindow = tk.Button(window, text="Quit", command=window.destroy)
quitWindow.place(x=525, y=35)

copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,)
copyWrite.tag_configure("superscript", offset=4)
copyWrite.insert("insert", "Developed by GK","", "TM", "superscript")
copyWrite.configure(state="disabled")
copyWrite.pack(side="top")
copyWrite.place(x=225, y=80)
 
window.mainloop()