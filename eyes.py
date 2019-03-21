import numpy as np
import pyscreenshot as sc
import cv2
import time


def see_amoeba_world(offsetX,offsetY,sizeX,sizeY): 
    last_time = time.time() 
    vision =  np.array(sc.grab(bbox=(offsetX,offsetY,sizeX,sizeY)))
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    cv2.imshow('window',cv2.cvtColor(vision, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    see_amoeba_world()

