import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
from sklearn import svm
import joblib
import time

#向量维度
VECTOR_SIZE = 3

#自定义数据队列
def queue_in(queue, data):
	ret = None
	if len(queue) >= VECTOR_SIZE:
		ret = queue.pop(0)
	queue.append(data)
	return ret, queue

#计算EAR的值
def eye_aspect_radio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


pwd = os.getcwd()
model_path = os.path.join(pwd, "model")
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

#导入训练好的模型
clf = joblib.load('ear_svm.m')

def begin():
	global VECTOR_SIZE
	#EAR阈值(EAR<0.3表示眨眼，大于0.3表示睁眼)
	EYE_AR_THRESH = 0.3
	#连续3帧EAR小于阈值时，表示眨眼动作发生
	EYE_AR_CONSEC_FRAMES = 3

	#人脸特征点对应的序号
	RIGHT_EYE_START = 37 - 1
	RIGHT_EYE_END = 42 - 1
	LEFT_EYE_START = 43 - 1
	LEFT_EYE_END = 48 - 1 

	frame_counter = 0
	blink_counter = 0
	ear_vector = []
	#读取摄像头获取的图像
	cap = cv2.VideoCapture(0)
	if(not cap.isOpened()):
		print('No camera is detected!!!')
		return
	while(cap.isOpened()):
		#计算程序运行一次的时间
		start = time.clock()
		ret, img = cap.read()

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		if not rects:
			cv2.putText(img, "No face detected!".format(blink_counter), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
		for rect in rects:
			print('-'*20)
			shape = predictor(gray, rect)
			#将人脸检测结果转换成numpy数组
			points = face_utils.shape_to_np(shape)

			leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
			rihgtEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
			leftEAR = eye_aspect_radio(leftEye)
			rightEAR = eye_aspect_radio(rihgtEye)
			print('leftEAR = {0}'.format(leftEAR))
			print('rightEAR = {0}'.format(rightEAR))

			ear = (leftEAR + rightEAR) / 2.0

			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rihgtEye)
			cv2.drawContours(img, [leftEyeHull], -1, (0, 252, 124), 1)
			cv2.drawContours(img ,[rightEyeHull], -1, (0, 252, 124), 1)

			#使用SVM向量机模型判断是否眨眼并计数
			ret, ear_vector = queue_in(ear_vector, ear)
			if(len(ear_vector) == VECTOR_SIZE):
				print(ear_vector)
				input_vector = []
				input_vector.append(ear_vector)
				res = clf.predict(input_vector)
				print(res)

				if res == 1:
					frame_counter += 1
				else:
					if frame_counter >= EYE_AR_CONSEC_FRAMES:
						blink_counter += 1
					frame_counter = 0

			end = (time.clock()-start)
			cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
			cv2.putText(img, "EAR:{:.2f}".format(ear), (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
			cv2.putText(img, "Timeuse:{:.2f}".format(end), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
			cv2.putText(img, "Press Q to exit!".format(end), (460, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)


		cv2.imshow('Frame', img)

		if cv2.waitKey(1)&0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()