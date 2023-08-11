from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import *
import os
import sys
import argparse
from PIL import Image
import cv2
import time

#----------------------Design of the Page----------------------

#Tab Title
st.set_page_config(page_title = "VISION", page_icon=":eyes:")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 340px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 340px;
        margin-left: -340px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#Title 
st.markdown("<h2 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Welcome to VISION!</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black; font-family: font of choice, fallback font no1, sans-serif;'>Visually Intelligent System for Identifying Objects in Nature</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: black; font-family: font of choice, fallback font no1, sans-serif;'>A project for CCS40 by Daguio, Diaz, and Tabual (BSIT-2A)</p>", unsafe_allow_html=True)
st.markdown('#') #Inserts empty space

#-----------------------------Paths-----------------------------

#If there are no uploaded image or video this block of code will call the dafault files.
EXAMPLE_VIDEO = os.path.join('data', 'videos', 'Cars.mp4')
EXAMPLE_IMG = os.path.join('data', 'images', 'Bus.jpg')

def get_sub_directories(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_sub_directories(os.path.join('yolov5','runs', 'detect')), key=os.path.getmtime)

#----------------------------Main Function----------------------------

def main():

    source = ("Image Detection", "Video Detection", "Live Feed Detection")
    index_of_source = st.sidebar.selectbox("Select Activity", range(
        len(source)), format_func = lambda x: source[x])
    
    #80 Objects that it can detect, this is from a yolov5.

    cocoClassesList = ["Person","Bicycle","Car","Motorcycle","Airplane","Bus","Train","Truck","Boat","Traffic Light","Fire Hydrant","Stop Sign","Parking Meter","Bench","Bird","Cat", \
    "Dog","Horse","Shee","Cow","Elephant","Bear","Zebra","Giraffe","Backpack","Umbrella","Handbag","Tie","Suitcase","Frisbee","Skis","Snowboard","Sports Ball","Kite","Baseball Bat",\
    "Baseball Glove","Skateboard","Surfboard","Tennis Racket","Bottle","Wine Glass","Cup","Fork","Knife","Spoon","Bowl","Banana","Apple","Sandwich","Orange","Broccoli","Carrot","Hot Dog",\
    "Pizza","Donut","Cake","Chair","Couch","Potted Plant","Bed","Dining Table","Toilet","TV","Laptop","Mouse","Remote","Keyboard","Cell Phone","Microwave","Oven","Toaster","Sink",\
    "Refrigerator","Book","Clock","Vase","Scissors","Teddy Bear","Hair Drier","Toothbrush", "All"]
    
    index_of_classes = st.sidebar.multiselect("Select Classes", range(
        len(cocoClassesList)), format_func = lambda x: cocoClassesList[x])
    
    selectedisAllinList = 80 in index_of_classes
    if selectedisAllinList == True:
        index_of_classes = index_of_classes.clear()
        
    print("Selected Classes: ", index_of_classes)
    
    #--------------------------Setting Parameters--------------------------

    deviceList = ['CPU']
    DEVICE = st.sidebar.selectbox("Select Devices", deviceList, index = 0)
    print("Devices: ", DEVICE)

    #For Confidence Level, the lower the confidence level the higher it can identify the objects.
    #For example, a significance threshold of 0.05 is equal to a 95% confidence threshold.

    MIN_CON_SCORE_THRES = st.sidebar.slider('Minimum Confidence Score Threshold', min_value = 0.0, max_value = 1.0, value = 0.4) 
    weights = os.path.join("weights", "yolov5s.pt") #Enables Yolov5 which is the main algorithm/package for object detection.

    #---------------Depending on the Selected Source/Activity-------------

    #Image Upload

    if index_of_source == 0:
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image", type = ['jpg','png', 'jpeg']) #Allowed Types
        
        #If there is an uploaded image
        
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text = 'Resource Loading...'):
                st.sidebar.text("Uploaded Image")
                st.sidebar.image(uploaded_file)
                image = Image.open(uploaded_file)
                image.save(os.path.join('data', 'images', uploaded_file.name))
                source_of_data = os.path.join('data', 'images', uploaded_file.name)

        #If there is no uploaded image

        elif uploaded_file is None:
            is_valid = True
            st.sidebar.text("Sample Image")
            st.sidebar.image(EXAMPLE_IMG)
            source_of_data = EXAMPLE_IMG
        
        else:
            is_valid = False
    
    #Video Upload

    elif index_of_source == 1:
        
        uploaded_file = st.sidebar.file_uploader("Upload Video", type = ['mp4'])
        
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text = 'Resource Loading...'):
                st.sidebar.text("Uploaded Video")
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                source_of_data = os.path.join("data", "videos", uploaded_file.name)
        
        elif uploaded_file is None:
            is_valid = True
            st.sidebar.text("Sample Video")
            st.sidebar.video(EXAMPLE_VIDEO)
            source_of_data = EXAMPLE_VIDEO
        
        else:
            is_valid = False
    
    #Live Feed

    else:
        
        selection_of_camera = st.sidebar.selectbox("Select Camera", ("Use Webcam", "Use Other Camera"), index = 0)
        if selection_of_camera:
            if selection_of_camera == "Use Other Camera":
                source_of_data = int(1)
                is_valid = True
            else:
                source_of_data = int(0)
                is_valid = True
        else:
            is_valid = False
        
        st.sidebar.markdown("<strong>To exit/clear Camera Window, please press 'Q' multiple times on Camera Window or 'Ctrl + C' on CMD. </strong>", unsafe_allow_html=True)
    
    #------------------------------Object Detection----------------------------

    if is_valid:
        print('valid')
        if st.button('Detect'):
            if index_of_classes:
                with st.spinner(text = 'Detecting, Please Wait...'):
                    run(weights = weights, 
                        source = source_of_data,  
                        conf_thres = MIN_CON_SCORE_THRES,
                        device = DEVICE,
                        save_txt = True,
                        save_conf = True,
                        classes = index_of_classes,
                        nosave = False, 
                        )
                        
            else:
                with st.spinner(text = 'Detecting, Please Wait...'):
                    run(weights = weights, 
                        source = source_of_data,  
                        conf_thres = MIN_CON_SCORE_THRES,
                        device = DEVICE,
                        save_txt = True,
                        save_conf = True,
                    nosave = False, 
                    )

            #Image Detection

            if index_of_source == 0:
                with st.spinner(text = 'Preparing Image'):
                    for img in os.listdir(get_detection_folder()):
                        if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png"):
                            path_of_Image = os.path.join(get_detection_folder(), img)
                            st.image(path_of_Image)
                    
                    st.markdown("### Output")
                    st.write("Path of Saved Image: ", path_of_Image)    
                    st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))  
                    st.balloons()
            
            #Video Detection
                    
            elif index_of_source == 1:
                with st.spinner(text = 'Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        if vid.endswith(".mp4"):
                            path_of_Video = os.path.join(get_detection_folder(), vid)
                            
                stframe = st.empty()
                capture = cv2.VideoCapture(path_of_Video)
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                print("Width: ", width, "\n")
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print("Height: ", height, "\n")

                while capture.isOpened():
                    ret, img = capture.read()
                    if ret:
                        stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                    else:
                        break
                
                capture.release()
                st.markdown("### Output")
                st.write("Path of Saved Video: ", path_of_Video)    
                st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))    
                st.balloons()

            #Live Feed Detection
            
            else:
                with st.spinner(text = 'Preparing Live Feed'):
                    for vid in os.listdir(get_detection_folder()):
                        if vid.endswith(".mp4"):
                            path_of_LiveFeed = os.path.join(get_detection_folder(), vid)
                    
                    st.markdown("### Output")
                    st.write("Path of Live Feed Saved Video: ", path_of_LiveFeed)    
                    st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))    
                    st.balloons()

# --------------------Calling the Main Function---------------------                                                                    
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
# ------------------------------------------------------------------


