import cv2
from tensorflow import keras
import numpy as np
import py_avataaars as pa

facec = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
model = keras.models.load_model("Model/model_v2.h5")
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
show_text=[0]

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_text[0] = maxindex
    
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
            avatar.render_png_file("Emoji_web/Angry.png")
            emoji = cv2.imread("Emoji_web/Angry.png")
        
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
            avatar.render_png_file("Emoji_web/Disgusted.png")
            emoji = cv2.imread("Emoji_web/Disgusted.png")

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
            avatar.render_png_file("Emoji_web/Fearful.png")
            emoji = cv2.imread("Emoji_web/Fearful.png")
        
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
            avatar.render_png_file("Emoji_web/Happy.png")
            emoji = cv2.imread("Emoji_web/Happy.png")
        
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
            avatar.render_png_file("Emoji_web/Neutral.png")
            emoji = cv2.imread("Emoji_web/Neutral.png")

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
            avatar.render_png_file("Emoji_web/Sad.png")
            emoji = cv2.imread("Emoji_web/Sad.png")

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
            avatar.render_png_file("Emoji_web/Suprised.png")
            emoji = cv2.imread("Emoji_web/Suprised.png")
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    
    def get_emoji(self):
        if show_text[0] == 0:
            emoji = cv2.imread("Emoji_web/Angry.png")
        if show_text[0] == 1:
            emoji = cv2.imread("Emoji_web/Disgusted.png")
        if show_text[0] == 2:
            emoji = cv2.imread("Emoji_web/Fearful.png")
        if show_text[0] == 3:
            emoji = cv2.imread("Emoji_web/Happy.png")
        if show_text[0] == 4:
            emoji = cv2.imread("Emoji_web/Neutral.png")
        if show_text[0] == 5:
            emoji = cv2.imread("Emoji_web/Sad.png")
        if show_text[0] == 6:
            emoji = cv2.imread("Emoji_web/Suprised.png")
        ret, jpeg = cv2.imencode('.jpg', emoji)
        return jpeg.tobytes()