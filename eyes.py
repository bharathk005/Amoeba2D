import numpy as np
from PIL import ImageGrab
import cv2
import time
from matplotlib import pyplot as plt
from imageai.Detection import ObjectDetection
import os
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

    


def see_amoeba_world(offsetX,offsetY,sizeX,sizeY): 
    last_time = time.time() 
    vision =  np.array(ImageGrab.grab(bbox=(offsetX,offsetY,sizeX,sizeY)))
    # visionbw = np.array(cv2.cvtColor(vision,cv2.COLOR_BGR2GRAY))
    # visionstack = np.dstack([visionbw]*3)
    detections, _ = detector.detectObjectsFromImage(input_type = 'array',output_type='array',input_image = vision)
    print('fps: {}'.format(1/(time.time()-last_time)))
    # # plt.imshow(np.array(detections))
    # # plt.show()
    cv2.imshow('image',detections)
    cv2.waitKey(1)


if __name__ == '__main__':
    while True:
        see_amoeba_world(120,250,920,720)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

