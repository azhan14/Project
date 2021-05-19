import numpy as np
import cv2
import os
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D

import py_avataaars as pa

emotion_model = keras.models.load_model("Model/model_v2.h5")

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
        

        cv2.imshow('Video', cv2.resize(frame,(600,480),interpolation = cv2.INTER_CUBIC))
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
        
        cv2.imshow('Emoji', cv2.resize(emoji,(480,480),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(2) & 0xFF == ord('w'):
            exit()

show_vid()
