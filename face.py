import streamlit as st
import cv2
#Making title and heading
st.title("Face Detector")
st.header("Upload image and detect face from it")
#Loading model
model = cv2.CascadeClassifier("C:/Users/Shahnawaz/Desktop/shanawaaz/haarcascade_frontalface_default.xml")
#Taking user image
user_img = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if user_img is not None:
    with open("Image1.jpg","wb") as f:
        f.write(user_img.read())

if user_img:   
    img_arr=cv2.imread("Image1.jpg")
    
    gray_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    faces = model.detectMultiScale(gray_img, minNeighbors=8, scaleFactor=1.1)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_arr, (x, y), (x+w, y+h), (0, 0, 256), 5)  
    
    st.image(img_arr, caption='Image with rectangle')