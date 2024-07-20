import cv2 as cv
import face_recognition as fr
import os
import time
from CaixaDelimitadora import CaixaDelimitadora

# Load known images and encode faces
DIR = os.path.dirname(os.path.abspath(__file__))
fotos_dir = os.path.join(DIR, 'fotos')

known_face_encodings = []
known_face_names = []

for filename in os.listdir(fotos_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
        img_path = os.path.join(fotos_dir, filename)
        img = fr.load_image_file(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        face_encodings = fr.face_encodings(img)
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])  # Use the file name without extension as the name

# Initialize webcam
video_capture = cv.VideoCapture(0)

# TEXTOS NAS IMAGENS
textFont = cv.FONT_HERSHEY_SIMPLEX
fontScale = 1
textColor = (255, 255, 255)  # White color in BGR
textThickness = 2

tempoFinal = 0
tempoInicial = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break
    tempoFinal = time.time()
    fps = int(1 / (tempoFinal - tempoInicial))
    tempoInicial = tempoFinal
    cv.putText(frame, f"FPS: {(fps)}", (5, 70), textFont, fontScale, textColor, 3)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face in the frame matches the known face(s)
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        # Use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        # Formata o texto com o nome e confiança
        confianca = (1 - face_distances[best_match_index]) * 100  # Calcula a confiança como um percentual
        texto = f"{name} - {round(confianca, 2)}%"
        (texto_largura, texto_altura), _ = cv.getTextSize(texto, textFont, fontScale, textThickness)  # Obtém a largura e altura do texto
        # Desenha o fundo do retângulo para o texto
        cv.rectangle(frame, (left, top - texto_altura - 10), (left + texto_largura, top), (0, 0, 0), -1)
        # Adiciona o texto ao frame
        cv.putText(frame, texto, (left, top - 10), textFont, fontScale, textColor, textThickness, cv.LINE_AA)
        
        # Draw a box around the face using CaixaDelimitadora
        caixaDelimitadora = CaixaDelimitadora(frame, (0, 255, 0))
        bbox = (left, top, right - left, bottom - top)
        frame = caixaDelimitadora.draw(bbox)

    # Display the resulting image
    cv.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv.destroyAllWindows()
