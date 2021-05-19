import numpy as np
from PIL import Image, ImageOps
import cv2
import streamlit as st
import py_avataaars as pa
import base64
from random import randrange
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
emotion_model = keras.models.load_model("Model/model_v2.h5")
font = cv2.FONT_HERSHEY_SIMPLEX 
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
show_text=[0]

def main():
    """Face Expression Detection App"""
    activities = ["Home","Detect your Facial expressions and mapped Emoji" ,"Create Customized Emoji"]
    choice = st.sidebar.selectbox("Select Activity",activities)
    if choice == 'Home':
        st.subheader("Welcome to Emojify")
        st.subheader(":smile: :worried: :fearful: :rage: :hushed:")

    if choice == 'Detect your Facial expressions and mapped Emoji':
        st.title("Webcam Live Feed")
        run = st.checkbox('Run')
        placeholder = st.empty()
        col1,col2 = st.beta_columns([2,1])
        with col1:
            FRAME_WINDOW = st.image([])
        with col2:
            FRAME_WINDOW1 = st.image([])
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            frame = cv2.resize(frame,(600,480))
            if not ret:
                break
            bounding_box = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
            if len(num_faces) == 0:
                placeholder.subheader("No Face Detected")
            else:
                placeholder.subheader(" ")

            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                show_text[0] = maxindex
            
            FRAME_WINDOW.image(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(600,480)))

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
            if len(num_faces) > 0:
                FRAME_WINDOW1.image(cv2.resize(cv2.cvtColor(emoji,cv2.COLOR_BGR2RGB),(480,480)))
            else:
                FRAME_WINDOW1.empty()

    
    if choice == 'Create Customized Emoji':
        st.sidebar.header('Customize your avatar')

        option_style = st.sidebar.selectbox('Style', ('CIRCLE', 'TRANSPARENT'))

        list_skin_color = ['TANNED','YELLOW','PALE','LIGHT','BROWN','DARK_BROWN','BLACK']
        list_top_type = ['NO_HAIR','EYE_PATCH','HAT','HIJAB','TURBAN',
                        'WINTER_HAT1','WINTER_HAT2','WINTER_HAT3',
                        'WINTER_HAT4','LONG_HAIR_BIG_HAIR','LONG_HAIR_BOB',
                        'LONG_HAIR_BUN','LONG_HAIR_CURLY','LONG_HAIR_CURVY',
                        'LONG_HAIR_DREADS','LONG_HAIR_FRIDA','LONG_HAIR_FRO',
                        'LONG_HAIR_FRO_BAND','LONG_HAIR_NOT_TOO_LONG',
                        'LONG_HAIR_SHAVED_SIDES','LONG_HAIR_MIA_WALLACE',
                        'LONG_HAIR_STRAIGHT','LONG_HAIR_STRAIGHT2',
                        'LONG_HAIR_STRAIGHT_STRAND','SHORT_HAIR_DREADS_01',
                        'SHORT_HAIR_DREADS_02','SHORT_HAIR_FRIZZLE',
                        'SHORT_HAIR_SHAGGY_MULLET','SHORT_HAIR_SHORT_CURLY',
                        'SHORT_HAIR_SHORT_FLAT','SHORT_HAIR_SHORT_ROUND',
                        'SHORT_HAIR_SHORT_WAVED','SHORT_HAIR_SIDES',
                        'SHORT_HAIR_THE_CAESAR','SHORT_HAIR_THE_CAESAR_SIDE_PART']
        list_hair_color = ['AUBURN','BLACK','BLONDE','BLONDE_GOLDEN','BROWN',
                        'BROWN_DARK','PASTEL_PINK','PLATINUM','RED','SILVER_GRAY']
        list_hat_color = ['BLACK','BLUE_01','BLUE_02','BLUE_03','GRAY_01','GRAY_02',
                        'HEATHER','PASTEL_BLUE','PASTEL_GREEN','PASTEL_ORANGE',
                        'PASTEL_RED','PASTEL_YELLOW','PINK','RED','WHITE']

        list_facial_hair_type = ['DEFAULT','BEARD_MEDIUM','BEARD_LIGHT','BEARD_MAJESTIC','MOUSTACHE_FANCY','MOUSTACHE_MAGNUM']
        list_facial_hair_color = ['AUBURN','BLACK','BLONDE','BLONDE_GOLDEN','BROWN','BROWN_DARK','PLATINUM','RED']
        list_mouth_type = ['DEFAULT','CONCERNED','DISBELIEF','EATING','GRIMACE','SAD','SCREAM_OPEN','SERIOUS','SMILE','TONGUE','TWINKLE','VOMIT']
        list_eye_type = ['DEFAULT','CLOSE','CRY','DIZZY','EYE_ROLL','HAPPY','HEARTS','SIDE','SQUINT','SURPRISED','WINK','WINK_WACKY']
        list_eyebrow_type = ['DEFAULT','DEFAULT_NATURAL','ANGRY','ANGRY_NATURAL','FLAT_NATURAL','RAISED_EXCITED','RAISED_EXCITED_NATURAL','SAD_CONCERNED','SAD_CONCERNED_NATURAL','UNI_BROW_NATURAL','UP_DOWN','UP_DOWN_NATURAL','FROWN_NATURAL']
        list_accessories_type = ['DEFAULT','KURT','PRESCRIPTION_01','PRESCRIPTION_02','ROUND','SUNGLASSES','WAYFARERS']
        list_Clothe_type = ['BLAZER_SHIRT','BLAZER_SWEATER','COLLAR_SWEATER','GRAPHIC_SHIRT','HOODIE','OVERALL','SHIRT_CREW_NECK','SHIRT_SCOOP_NECK','SHIRT_V_NECK']
        list_Clothe_color = ['BLACK','BLUE_01','BLUE_02','BLUE_03','GRAY_01','GRAY_02','HEATHER','PASTEL_BLUE','PASTEL_GREEN','PASTEL_ORANGE','PASTEL_RED','PASTEL_YELLOW','PINK','RED','WHITE']
        list_Clothe_graphic_type = ['BAT','CUMBIA','DEER','DIAMOND','HOLA','PIZZA','RESIST','SELENA','BEAR','SKULL_OUTLINE','SKULL']

        # if st.button('Random Avatar'):
        #     index_skin_color = randrange(0, len(list_skin_color) )
        #     index_top_type = randrange(0, len(list_top_type) )
        #     index_hair_color = randrange(0, len(list_hair_color) )
        #     index_hat_color = randrange(0, len(list_hat_color) )
        #     index_facial_hair_type = randrange(0, len(list_facial_hair_type) )
        #     index_facial_hair_color= randrange(0, len(list_facial_hair_color) )
        #     index_mouth_type = randrange(0, len(list_mouth_type) )
        #     index_eye_type = randrange(0, len(list_eye_type) )
        #     index_eyebrow_type = randrange(0, len(list_eyebrow_type) )
        #     index_accessories_type = randrange(0, len(list_accessories_type) )
        #     index_Clothe_type = randrange(0, len(list_Clothe_type) )
        #     index_Clothe_color = randrange(0, len(list_Clothe_color) )
        #     index_Clothe_graphic_type = randrange(0, len(list_Clothe_graphic_type) )
        # else:
        index_skin_color = 0
        index_top_type = 0
        index_hair_color = 0
        index_hat_color = 0
        index_facial_hair_type = 0
        index_facial_hair_color = 0
        index_mouth_type = 0
        index_eye_type = 0
        index_eyebrow_type = 0
        index_accessories_type = 0
        index_Clothe_type = 0
        index_Clothe_color = 0
        index_Clothe_graphic_type = 0

        option_skin_color = st.sidebar.selectbox('Skin color',
                                                list_skin_color,
                                                index = index_skin_color )

        st.sidebar.subheader('Head top')
        option_top_type = st.sidebar.selectbox('Head top',
                                                list_top_type,
                                                index = index_top_type)
        option_hair_color = st.sidebar.selectbox('Hair color',
                                                list_hair_color,
                                                index = index_hair_color)
        option_hat_color = st.sidebar.selectbox('Hat color',
                                                list_hat_color,
                                                index = index_hat_color)

        st.sidebar.subheader('Face')
        option_facial_hair_type = st.sidebar.selectbox('Facial hair type',
                                                        list_facial_hair_type,
                                                        index = index_facial_hair_type)
        option_facial_hair_color = st.sidebar.selectbox('Facial hair color',
                                                        list_facial_hair_color,
                                                        index = index_facial_hair_color)
        option_mouth_type = st.sidebar.selectbox('Mouth type',
                                                list_mouth_type,
                                                index = index_mouth_type)
        option_eye_type = st.sidebar.selectbox('Eye type',
                                                list_eye_type,
                                                index = index_eye_type)
        option_eyebrow_type = st.sidebar.selectbox('Eyebrow type',
                                                    list_eyebrow_type,
                                                    index = index_eyebrow_type)

        st.sidebar.subheader('Clothe and accessories')
        option_accessories_type = st.sidebar.selectbox('Accessories type',
                                                        list_accessories_type,
                                                        index = index_accessories_type)
        option_Clothe_type = st.sidebar.selectbox('Clothe type',
                                                list_Clothe_type,
                                                index = index_Clothe_type)
        option_Clothe_color = st.sidebar.selectbox('Clothe Color',
                                                    list_Clothe_color,
                                                    index = index_Clothe_color)
        option_Clothe_graphic_type = st.sidebar.selectbox('Clothe graphic type',
                                                        list_Clothe_graphic_type,
                                                        index = index_Clothe_graphic_type)


        avatar = pa.PyAvataaar(
            # style=pa.AvatarStyle.CIRCLE,
            style=eval('pa.AvatarStyle.%s' % option_style),
            skin_color=eval('pa.SkinColor.%s' % option_skin_color),
            top_type=eval('pa.TopType.SHORT_HAIR_SHORT_FLAT.%s' % option_top_type),
            hair_color=eval('pa.HairColor.%s' % option_hair_color),
            hat_color=eval('pa.Color.%s' % option_hat_color),
            facial_hair_type=eval('pa.FacialHairType.%s' % option_facial_hair_type),
            facial_hair_color=eval('pa.HairColor.%s' % option_facial_hair_color),
            mouth_type=eval('pa.MouthType.%s' % option_mouth_type),
            eye_type=eval('pa.EyesType.%s' % option_eye_type),
            eyebrow_type=eval('pa.EyebrowType.%s' % option_eyebrow_type),
            nose_type=pa.NoseType.DEFAULT,
            accessories_type=eval('pa.AccessoriesType.%s' % option_accessories_type),
            clothe_type=eval('pa.ClotheType.%s' % option_Clothe_type),
            clothe_color=eval('pa.Color.%s' % option_Clothe_color),
            clothe_graphic_type=eval('pa.ClotheGraphicType.%s' %option_Clothe_graphic_type)
        )

        # Custom function for encoding and downloading avatar image
        def imagedownload(filename):
            image_file = open(filename, 'rb')
            b64 = base64.b64encode(image_file.read()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
            return href

        st.subheader('**Rendered Avatar**')
        rendered_avatar = avatar.render_png_file('avatar.png')
        image = Image.open('avatar.png')
        st.image(image)
        st.markdown(imagedownload('avatar.png'), unsafe_allow_html=True)
main()