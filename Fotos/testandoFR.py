import cv2
import face_recognition as fr

imgElon = fr.load_image_file('Elon.jpg')#Carrega Imagem
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgElonTest = fr.load_image_file('ElonTest.jpg')
imgElonTest = cv2.cvtColor(imgElonTest,cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgElon)[0]#Localiza a face | A list of tuples of found face locations in css (top, right, bottom, left) order
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)

encodeElon = fr.face_encodings(imgElon)[0] #Return a list of 128-dimensional face encodings (one for each face in the image)
encodeElonTest = fr.face_encodings(imgElonTest)[0] #Return a list of 128-dimensional face encodings (one for each face in the image)

comparacao = fr.compare_faces([encodeElon],encodeElonTest)#Return a list of 128-dimensional face encodings (one for each face in the image)
distancia = fr.face_distance([encodeElon],encodeElonTest)#A numpy ndarray with the distance for each face in the same order as the 'faces' array

print(comparacao,distancia)
cv2.imshow('Elon',imgElon)
cv2.imshow('Elon Test',imgElonTest)
cv2.waitKey(0)