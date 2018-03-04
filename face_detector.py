#!/usr/bin/python3 
# -*- coding: utf-8 -*-

import dlib

face_detector = dlib.get_frontal_face_detector()

face_predictor_path = './shape_predictor_68_face_landmarks.dat'
face_predictor = dlib.shape_predictor(face_predictor_path)