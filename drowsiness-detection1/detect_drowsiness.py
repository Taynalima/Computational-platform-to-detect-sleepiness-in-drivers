#TERMINAL
#  python3 detect_drowsiness.py --shape_predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav --webcam 0

# importar pacotes
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

#-*- Coding: UTF-8 -*-
#coding: utf-8

def sound_alarm(path):
	# toca o som de alarme
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# calcula as distâncias euclidianas entre os dois conjuntos de
	# marcos de olho verticais (x, y) -coordenados
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# calcula a distância euclidiana entre a coordenada horizontal do marco do olho (x, y)
	C = dist.euclidean(eye[0], eye[3])

	# calcula a proporção do olho
	ear = (A + B) / (2.0 * C)

	# retorna a proporção do olho
	return ear
 
# constrói o argumento e analisa-os

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())

# define duas constantes, uma para a proporção do olho para indicar
# piscada e, em seguida, uma segunda constante para o número de landmarks
# O olho deve estar abaixo do limite para desencadear o alarme
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48

# inicializa o landmark, bem como um boolean usado para
# indicar se o alarme está disparando
COUNTER = 0
ALARM_ON = False

# inicializa o detector de faces do dlib (baseado em HOG) e, em seguida, cria
# o landmark predictor
print("[INFO] Carregando detector de sonolencia...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# pega os índices dos pontos de referência faciais para a esquerda e
# olho direito, respectivamente
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# inicia o encadeamento de fluxo de vídeo
print("[INFO] Iniciando fluxo de vídeo...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop sobre quadros do fluxo de vídeo
while True:
	# pega o quadro do fluxo de arquivos de vídeo encadeados, redimensiona
	# , e converte-os em escala de cinza
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detecta rostos no quadro de tons de cinza
	rects = detector(gray, 0)

	# loop sobre as detecções de rosto
	for rect in rects:
		# determina os marcos faciais para a região da face e, em seguida,
		# converte o ponto de referência facial (x, y) -coordena para um NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extrai as coordenadas do olho esquerdo e direito, depois usa as
		# coordenadas para calcular a proporção do olho para ambos os olhos
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# faz a media da proporção do olho em conjunto para ambos os olhos
		ear = (leftEAR + rightEAR) / 2.0

		# calcula o casco convexo para o olho esquerdo e direito, então
		# visualiza cada um dos olhos
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# verifica se a proporção do olho está abaixo do piscar de olhos
		# threshold, e se for o caso, incrementa o contador de frames
		if ear < EYE_AR_THRESH:
			COUNTER += 1.3

			# se os olhos estiverem fechados por um número suficiente de tempo
			# então soa o alarme
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# se o alarme não estiver ligado, liga
				if not ALARM_ON:
					ALARM_ON = True

					# verifica para ver se um arquivo de alarme foi fornecido,
					# e se assim for, inicia uma thread para ter o som de alarme
					# tocado em segundo plano
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# escreve um alerta de olhos fechados na tela
				cv2.putText(frame, "ALERTA, OLHOS FECHADOS!", (5, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 128), 2)

		# caso contrário, a proporção do olho não está abaixo do piscar de olhos
		# limiar, então redefine o contador e não soa o alarme
		else:
			COUNTER = 0
			ALARM_ON = False

		# escreve na tela a relação de aspecto do olho computado no quadro para ajudar
		# com depuração e definição da proporção correta do olho
		# limiares e landmarks
		cv2.putText(frame, "DIMENSAO DO OLHO: {:.2f}".format(ear), (5, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 128), 2)
 
	#mostra o quadro
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# se a tecla `q` for pressionada, interrompe o loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
