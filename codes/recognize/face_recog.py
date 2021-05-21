# To capture face recognition using a VIDEO file.
import os 
from cv2 import cv2
import face_recognition

# Import your video file
video_file = cv2.VideoCapture(os.path.abspath("recognize/videos/input.mp4"))

# Capture the length based on the frame.
length = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))

# We need to add all the faces that we want our code to recognize
image_J_sankar = face_recognition.load_image_file(os.path.abspath("recognize/images/J_sankar.jpg"))
image_saptarshi_mukherjee = face_recognition.load_image_file(os.path.abspath("recognize/images/saptarshi_mukherjee.jpg"))
image_sebastian_wuster = face_recognition.load_image_file(os.path.abspath("recognize/images/sebastian_wuster.jpg"))
image_snighdha_thakur = face_recognition.load_image_file(os.path.abspath("recognize/images/snighdha_thakur.jpg"))
image_abhijeet = face_recognition.load_image_file(os.path.abspath("recognize/images/Abhijeet_Patra.jpg"))
image_anandeertha = face_recognition.load_image_file(os.path.abspath("recognize/images/anandeertha_mangasuli.jpg"))
image_bhargav = face_recognition.load_image_file(os.path.abspath("recognize/images/bhargav_ram.png"))
image_kavin = face_recognition.load_image_file(os.path.abspath("recognize/images/NR_Kavin_kumar.jpg"))


# Generate the face encoding for the image that has been passed.
J_sankar_face_1 = face_recognition.face_encodings(image_J_sankar)[0]
saptarshi_mukherjee_face_1 = face_recognition.face_encodings(image_saptarshi_mukherjee)[0]
sebastian_wuster_face_1 = face_recognition.face_encodings(image_sebastian_wuster)[0]
snighdha_thakur_face_1 = face_recognition.face_encodings(image_snighdha_thakur)[0]
abhijeet_face_1 = face_recognition.face_encodings(image_abhijeet)[0]
anandeertha_face_1 = face_recognition.face_encodings(image_anandeertha)[0]
bhargav_face_1 = face_recognition.face_encodings(image_bhargav)[0]
kavin_face_1 = face_recognition.face_encodings(image_kavin)[0]


# Make a list of all the known faces that we want to be recognized based on the 
# encoding.
known_faces = [
J_sankar_face_1,saptarshi_mukherjee_face_1,sebastian_wuster_face_1,snighdha_thakur_face_1,
abhijeet_face_1,anandeertha_face_1,bhargav_face_1,kavin_face_1
]

facial_points = []
face_encodings = []
facial_number = 0

while True:
    return_value, frame = video_file.read()
    facial_number = facial_number + 1
    
    if not return_value:
        break
    rgb_frame = frame[:, :, ::-1]

    facial_points = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, facial_points)

    facial_names = []
    for encoding in face_encodings:
        match = face_recognition.compare_faces(known_faces, encoding, tolerance=0.50)
        

        name = ""
        if match[0]:
            name = "J_sankar"
        if match[1]:
            name = "saptarshi_mukherjee"
        if match[2]:
            name = "sebastian_wuster"
        if match[3]:
            name = "snighdha_thakur"
        if match[4]:
            name = "Abhijeet Patra"   
        if match[5]:
            name = "A. Mangasuli "
        if match[6]:
            name = "Bhargav Ram"
        if match[7]:
            name = "Kavin Kumar"

        
        facial_names.append(name)

    for (top, right, bottom, left), name in zip(facial_points, facial_names):
        # Enclose the face with the box - Red color 
        # top, right, bottom, left - 129, 710, 373, 465
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Name the characters in the Box created above
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    codec = int(video_file.get(cv2.CAP_PROP_FOURCC))
    fps = int(video_file.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_movie = cv2.VideoWriter("output/output_{}.mp4".format(facial_number), codec, fps, (frame_width,frame_height))
    print("Writing frame {} / {}".format(facial_number, length))
    output_movie.write(frame)

video_file.release()
output_movie.release()
cv2.destroyAllWindows()
