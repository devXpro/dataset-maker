import cv2
import tensorflow as tf
import numpy as np
from time import time 
import os

def preprocessing(img):
    return img/127.5 - 1


PIC_PATH = 'rec6.jpg'
# shape for result image
PIC_H = 520
PIC_W = 300

MODEL_PATH = 'sq224.tflite'
DATA_DIR = 'photo'
MODEL_SHAPE = 224
PROBABILITY_LIMIT = 0.15 # recommended 0.1-0.5
# then smaller limit then fatter lines


interpreter = tf.lite.Interpreter(model_path=MODEL_PATH) 
interpreter.allocate_tensors() 
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


frame = cv2.imread(PIC_PATH)


start = time()

frame_copy = frame
frame = cv2.resize(frame, (MODEL_SHAPE, MODEL_SHAPE))
proc_image = preprocessing(frame)
proc_image = np.expand_dims(proc_image, axis=0).astype(np.float32)

interpreter.set_tensor(input_index, proc_image)
interpreter.invoke()

prediction = (interpreter.get_tensor(output_index)>PROBABILITY_LIMIT)*255
prediction = cv2.cvtColor(np.float32(prediction.squeeze()),cv2.COLOR_GRAY2RGB).astype('uint8')
prediction[:,:,1] = 0


cv2.imwrite(os.path.join(DATA_DIR,f'SINGLE.jpg'), prediction)

frame = cv2.resize(frame_copy, (PIC_W, PIC_H))
prediction = cv2.resize(prediction, (PIC_W, PIC_H))
overlap = cv2.addWeighted(frame,0.8,prediction,1, 0)
concatenate = np.concatenate((frame, prediction, overlap), axis=1)
cv2.imwrite(os.path.join(DATA_DIR,f'SINGLE_con.jpg'), concatenate)


cv2.imshow('Frame', concatenate)

print(start - time())
cv2.waitKey(5000)

