import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, ttk
import pickle
import PIL.Image
import PIL.ImageTk
from skimage.transform import resize
from skimage.io import imread

datadir = 'data'
Categories = os.listdir(datadir)

root = Tk()
root.title("CHRONIC Detection")
root.geometry("800x600")
root.configure(bg='#D3D3D3')

value = StringVar()
panel = Label(root)

# Load the three models
filename1 = 'model_knn.sav'
loaded_model1 = pickle.load(open(filename1, 'rb'))

filename2 = 'model_rf.sav'
loaded_model2 = pickle.load(open(filename2, 'rb'))

filename3 = 'model_svm.sav'
loaded_model3 = pickle.load(open(filename3, 'rb'))

def detect(filename):
    img = imread(filename)
    plt.imshow(img)
    plt.show()
    img_resize = resize(img, (50, 50, 3))
    l = [img_resize.flatten()]

    # Make predictions using the three models
    probability1 = loaded_model1.predict(l)
    probability2 = loaded_model2.predict(l)
    probability3 = loaded_model3.predict(l)

    # Calculate the mean prediction value
    mean_prediction = np.mean([probability1, probability2, probability3])

    print("Model 1 Prediction:", int(probability1))
    print("Model 2 Prediction:", int(probability2))
    print("Model 3 Prediction:", int(probability3))
    print("Mean Prediction Value:", mean_prediction)

    # Display the results
    result_text = " Fused Prediction: " + Categories[int(mean_prediction)] + "\n\n   Predictions:\n"
    result_text += "KNN " + Categories[int(probability1)] + "\n"
    result_text += "RF: " + Categories[int(probability2)] + "\n"
    result_text += "SVM : " + Categories[int(probability3)]

    print(result_text)
    value.set(result_text)

def ClickAction(event=None):
    filename = filedialog.askopenfilename()
    img = PIL.Image.open(filename)
    img = img.resize((250, 250))
    img = PIL.ImageTk.PhotoImage(img)
    global panel
    panel = Label(root, image=img)
    panel.image = img
    panel = panel.place(relx=0.5, rely=0.2)
    detect(filename)

canvas = Canvas(root, width=800, height=100, bg="Blue")
canvas.place(relx=0, rely=0)

button = Button(root, text='Load Image', font=(None, 12), activeforeground='red', bd=10, bg='red', relief=RAISED, command=ClickAction)
button.place(relx=0.4, rely=0.9)

result_label = Label(root, textvariable=value, font=(None, 12), justify=LEFT)
result_label.place(relx=0.1, rely=0.5)

root.mainloop()
