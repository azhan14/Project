import numpy as np
import cv2
# from PIL import Image as img
# from PIL import ImageTk
# import tkinter as tk
# from tkinter import *
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D

import py_avataaars as pa

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('Model/model_Weight.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]
def show_vid():
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame,(600,500))
        if not ret:
            break
        bounding_box = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_text[0] = maxindex
        
        # if ret is None:
        #     print("Error")
        # elif ret:
        #     global last_frame1
        #     last_frame1 = frame.copy()
        #     pic = cv2.cvtColor(last_frame1,cv2.COLOR_BGR2RGB)
        #     img1 =img.fromarray(pic)
        #     imgtk = ImageTk.PhotoImage(image=img1)
        #     lmain.imgtk = imgtk
        #     lmain.configure(image=imgtk)
        #     lmain.after(10,show_vid) 

        cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

        if show_text[0] == 0:
            avatar = pa.PyAvataaar(
            style=pa.AvatarStyle.CIRCLE,
            skin_color=pa.SkinColor.LIGHT,
            hair_color=pa.HairColor.BROWN,
            facial_hair_type=pa.FacialHairType.DEFAULT,
            facial_hair_color=pa.HairColor.BLACK,
            top_type=pa.TopType.SHORT_HAIR_SHORT_FLAT,
            hat_color=pa.Color.BLACK,
            mouth_type=pa.MouthType.SERIOUS,
            eye_type=pa.EyesType.SQUINT,
            eyebrow_type=pa.EyebrowType.ANGRY,
            nose_type=pa.NoseType.DEFAULT,
            accessories_type=pa.AccessoriesType.DEFAULT,
            clothe_type=pa.ClotheType.GRAPHIC_SHIRT,
            clothe_color=pa.Color.HEATHER,
            clothe_graphic_type=pa.ClotheGraphicType.BAT,)
            avatar.render_png_file("Emoji/Angry.png")
            emoji = cv2.imread("Emoji/Angry.png")
        
        if show_text[0] == 1:
            avatar = pa.PyAvataaar(
            style=pa.AvatarStyle.CIRCLE,
            skin_color=pa.SkinColor.LIGHT,
            hair_color=pa.HairColor.BROWN,
            facial_hair_type=pa.FacialHairType.DEFAULT,
            facial_hair_color=pa.HairColor.BLACK,
            top_type=pa.TopType.SHORT_HAIR_SHORT_FLAT,
            hat_color=pa.Color.BLACK,
            mouth_type=pa.MouthType.VOMIT,
            eye_type=pa.EyesType.SQUINT,
            eyebrow_type=pa.EyebrowType.FROWN_NATURAL,
            nose_type=pa.NoseType.DEFAULT,
            accessories_type=pa.AccessoriesType.DEFAULT,
            clothe_type=pa.ClotheType.GRAPHIC_SHIRT,
            clothe_color=pa.Color.HEATHER,
            clothe_graphic_type=pa.ClotheGraphicType.BAT,)
            avatar.render_png_file("Emoji/Disgusted.png")
            emoji = cv2.imread("Emoji/Disgusted.png")

        if show_text[0] == 2:
            avatar = pa.PyAvataaar(
            style=pa.AvatarStyle.CIRCLE,
            skin_color=pa.SkinColor.LIGHT,
            hair_color=pa.HairColor.BROWN,
            facial_hair_type=pa.FacialHairType.DEFAULT,
            facial_hair_color=pa.HairColor.BLACK,
            top_type=pa.TopType.SHORT_HAIR_SHORT_FLAT,
            hat_color=pa.Color.BLACK,
            mouth_type=pa.MouthType.SCREAM_OPEN,
            eye_type=pa.EyesType.SURPRISED,
            eyebrow_type=pa.EyebrowType.RAISED_EXCITED,
            nose_type=pa.NoseType.DEFAULT,
            accessories_type=pa.AccessoriesType.DEFAULT,
            clothe_type=pa.ClotheType.GRAPHIC_SHIRT,
            clothe_color=pa.Color.HEATHER,
            clothe_graphic_type=pa.ClotheGraphicType.BAT,)
            avatar.render_png_file("Emoji/Fearful.png")
            emoji = cv2.imread("Emoji/Fearful.png")
        
        if show_text[0] == 3:
            avatar = pa.PyAvataaar(
            style=pa.AvatarStyle.CIRCLE,
            skin_color=pa.SkinColor.LIGHT,
            hair_color=pa.HairColor.BROWN,
            facial_hair_type=pa.FacialHairType.DEFAULT,
            facial_hair_color=pa.HairColor.BLACK,
            top_type=pa.TopType.SHORT_HAIR_SHORT_FLAT,
            hat_color=pa.Color.BLACK,
            mouth_type=pa.MouthType.SMILE,
            eye_type=pa.EyesType.SQUINT,
            eyebrow_type=pa.EyebrowType.DEFAULT,
            nose_type=pa.NoseType.DEFAULT,
            accessories_type=pa.AccessoriesType.DEFAULT,
            clothe_type=pa.ClotheType.GRAPHIC_SHIRT,
            clothe_color=pa.Color.HEATHER,
            clothe_graphic_type=pa.ClotheGraphicType.BAT,)
            avatar.render_png_file("Emoji/Happy.png")
            emoji = cv2.imread("Emoji/Happy.png")
        
        if show_text[0] == 4:
            avatar = pa.PyAvataaar(
            style=pa.AvatarStyle.CIRCLE,
            skin_color=pa.SkinColor.LIGHT,
            hair_color=pa.HairColor.BROWN,
            facial_hair_type=pa.FacialHairType.DEFAULT,
            facial_hair_color=pa.HairColor.BLACK,
            top_type=pa.TopType.SHORT_HAIR_SHORT_FLAT,
            hat_color=pa.Color.BLACK,
            mouth_type=pa.MouthType.SERIOUS,
            eye_type=pa.EyesType.DEFAULT,
            eyebrow_type=pa.EyebrowType.DEFAULT,
            nose_type=pa.NoseType.DEFAULT,
            accessories_type=pa.AccessoriesType.DEFAULT,
            clothe_type=pa.ClotheType.GRAPHIC_SHIRT,
            clothe_color=pa.Color.HEATHER,
            clothe_graphic_type=pa.ClotheGraphicType.BAT,)
            avatar.render_png_file("Emoji/Neutral.png")
            emoji = cv2.imread("Emoji/Neutral.png")

        if show_text[0] == 5:
            avatar = pa.PyAvataaar(
            style=pa.AvatarStyle.CIRCLE,
            skin_color=pa.SkinColor.LIGHT,
            hair_color=pa.HairColor.BROWN,
            facial_hair_type=pa.FacialHairType.DEFAULT,
            facial_hair_color=pa.HairColor.BLACK,
            top_type=pa.TopType.SHORT_HAIR_SHORT_FLAT,
            hat_color=pa.Color.BLACK,
            mouth_type=pa.MouthType.SAD,
            eye_type=pa.EyesType.CRY,
            eyebrow_type=pa.EyebrowType.SAD_CONCERNED_NATURAL,
            nose_type=pa.NoseType.DEFAULT,
            accessories_type=pa.AccessoriesType.DEFAULT,
            clothe_type=pa.ClotheType.GRAPHIC_SHIRT,
            clothe_color=pa.Color.HEATHER,
            clothe_graphic_type=pa.ClotheGraphicType.BAT,)
            avatar.render_png_file("Emoji/Sad.png")
            emoji = cv2.imread("Emoji/Sad.png")

        if show_text[0] == 6:
            avatar = pa.PyAvataaar(
            style=pa.AvatarStyle.CIRCLE,
            skin_color=pa.SkinColor.LIGHT,
            hair_color=pa.HairColor.BROWN,
            facial_hair_type=pa.FacialHairType.DEFAULT,
            facial_hair_color=pa.HairColor.BLACK,
            top_type=pa.TopType.SHORT_HAIR_SHORT_FLAT,
            hat_color=pa.Color.BLACK,
            mouth_type=pa.MouthType.DISBELIEF,
            eye_type=pa.EyesType.SURPRISED,
            eyebrow_type=pa.EyebrowType.RAISED_EXCITED_NATURAL,
            nose_type=pa.NoseType.DEFAULT,
            accessories_type=pa.AccessoriesType.DEFAULT,
            clothe_type=pa.ClotheType.GRAPHIC_SHIRT,
            clothe_color=pa.Color.HEATHER,
            clothe_graphic_type=pa.ClotheGraphicType.BAT,)
            avatar.render_png_file("Emoji/Suprised.png")
            emoji = cv2.imread("Emoji/Suprised.png")
        
        cv2.imshow('Emoji', cv2.resize(emoji,(600,600),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(2) & 0xFF == ord('w'):
            exit()

    # pic2=cv2.cvtColor(emoji,cv2.COLOR_BGR2RGB)
    # img2=img.fromarray(frame2)
    # imgtk2=ImageTk.PhotoImage(image=img2)
    # lmain2.imgtk2=imgtk2
    # lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
    
    # lmain2.configure(image=imgtk2)
    # lmain2.after(10, show_vid2)


# if __name__ == '__main__':
#     root=tk.Tk()   
#     heading2=Label(root,text="Photo to Emoji",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')                                 
    
#     heading2.pack()
#     lmain = tk.Label(master=root,padx=50,bd=10)
#     lmain2 = tk.Label(master=root,bd=10)

#     lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
#     lmain.pack(side=LEFT)
#     lmain.place(x=50,y=250)
#     lmain3.pack()
#     lmain3.place(x=960,y=250)
#     lmain2.pack(side=RIGHT)
#     lmain2.place(x=900,y=350)

#     root.title("Photo To Emoji")            
#     root.geometry("1400x900+100+10") 
#     root['bg']='black'
#     exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
#     show_vid()
#     show_vid2()
#     root.mainloop()

# while True:
show_vid()
