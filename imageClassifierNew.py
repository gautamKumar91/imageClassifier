from tkinter import * #Comes Defalut With Python3
from tkinter import filedialog as fd
from tkinter import messagebox as ms
import tkinter as tk
from PIL import ImageTk, Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
 
#Main Class Object
class application:
    def __init__(self,master):
        self.master = master
        self.master.geometry('375x200')
        self.c_size = (100,100)
        self.setup_gui(self.c_size)
        self.img=None
        self.model = VGG16()
 
    def setup_gui(self,s):        
        self.canvas = tk.Canvas(self.master,height=s[1],width=s[0],
            bg='black',bd=10,relief='ridge')
        self.canvas.place(x=0, y=0)                
        tk.Button(text='Open New Image',bd=2,fg='white',bg='black',font=('',10)
            ,command=self.make_image).place(x=5, y=130)   
        tk.Button(text='Classify Image',bd=2,fg='white',bg='black',font=('',10)
            ,command=self.classify_image).place(x=120, y=130)  
        tk.Button(text='Clear',bd=2,fg='white',bg='black',font=('',10)
            ,command=self.clear).place(x=220, y=130)  
        tk.Button(text='Quit',bd=2,fg='white',bg='black',font=('',10)
            ,command=self.master.destroy).place(x=265, y=130)  
        
        self.var1 = tk.StringVar()
        self.var2 = tk.StringVar()
        self.var3 = tk.StringVar()
        
        self.message1=tk.Label(self.master,textvariable=self.var1,bg='gray',
           font=('Ubuntu',10),bd=2,fg='black',relief='sunken',anchor=tk.W).place(x=130,y=20)
        self.message2=tk.Label(self.master,textvariable=self.var2,bg='gray',
           font=('Ubuntu',10),bd=2,fg='black',relief='sunken',anchor=tk.W).place(x=130,y=45)
        self.status=tk.Label(self.master,textvariable=self.var3,bg='gray',
            font=('Ubuntu',10),bd=2,fg='black',relief='sunken').place(x=5,y=170)
        
        self.var1.set("Predicted Value: ")
        self.var2.set("Predicted Score: ")
        self.var3.set("Current Image: ")
        
    def clear(self):
        self.var1.set("Predicted Value: ")
        self.var2.set("Predicted Score: ")
        self.var3.set("Current Image: ")
        self.canvas.delete(tk.ALL)
 
    def make_image(self):
        try:
            self.File = fd.askopenfilename()            
            self.pilImage = Image.open(self.File)
            re=self.pilImage.resize((100,100),Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(re)
            self.canvas.delete(tk.ALL)
            self.canvas.create_image(self.c_size[0]/2+10,self.c_size[1]/2+10,
                anchor=tk.CENTER,image=self.img)
            self.status1 = 'Current Image: '+ str(self.File).split("/")[-1]
            self.var3.set(self.status1)
        except:
            ms.showerror('Error!','File type is unsupported.')
            
    def classify_image(self):
        imagePath = str(self.File)        
        print(imagePath)
        #imagePath = imagePath.split(":",maxsplit=1)[1]
        #print(imagePath)
        try:
            image = load_img(imagePath, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            # predict the probability across all output classes
            yhat = self.model.predict(image)
            # convert the probabilities to class labels
            label = decode_predictions(yhat)
            # retrieve the most likely result, e.g. highest probability
            label = label[0][0]
            
            self.message1 = 'Predicted Value: '+  str(label[1])
            self.message2 ='Predicted Score: '+ str(round(label[2],2))
            
            self.var1.set(self.message1)
            self.var2.set(self.message2)
            
            #self.message1.configure(text = res1)
            #self.message2.configure(text = res2)
        except:
            ms.showerror('Error!','File type is unsupported.')
 
#creating object of class and tk window-
root=tk.Tk()
root.configure(bg='white')
root.title('VGG16 Image Classification ToolBox')
application(root)
root.mainloop()