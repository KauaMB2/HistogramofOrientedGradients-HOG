import cv2 as cv
import face_recognition as fr
import os
import time

class CaixaDelimitadora:
    def __init__(self, frame, cor):
        self.__frame = frame
        self.__cor = cor

    def draw(self, bbox, l=30, t=10):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        #Rectangle
        cv.rectangle(self.__frame, (x, y), (x + w, y + h), self.__cor, thickness=4)  # Draw the bounding box
        # Top Left
        cv.line(self.__frame, (x, y), (x + l, y), self.__cor, t)
        cv.line(self.__frame, (x, y), (x, y + l), self.__cor, t)
        # Top Right
        cv.line(self.__frame, (x1, y), (x1 - l, y), self.__cor, t)
        cv.line(self.__frame, (x1, y), (x1, y + l), self.__cor, t)
        return self.__frame

    