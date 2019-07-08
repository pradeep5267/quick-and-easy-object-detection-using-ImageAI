#%%
from imageai.Detection import ObjectDetection
import os
import matplotlib.pyplot as plt
import sys
#%%
execution_path = os.getcwd()
print(execution_path)

#%%
try:
    img_fn = sys.argv[1]
except:
    img_fn = './test3.jpg'

img = plt.imread(img_fn)
if img is None:
    print ('Failed to load image file:', img_fn)
    sys.exit(1)

#%%
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=img, output_image_path=os.path.join(execution_path , "imagenew.jpg"))

#%%
for eachObject in detections:
    print(eachObject)
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")