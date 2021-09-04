import cv2 as cv
import os as o
import numpy as np
from time import time

dataruta='C:/Users/hp/Documents/Python/Analisis Imagenes/Reconocimiento Facial/Data'
listadata=o.listdir(dataruta)

ids=[]
rostrosdata=[]
id=0

tiempoinicial=time()
for f in listadata:
    rutacompleta=dataruta+'/'+ f
    print('iniciando Lectura')
    for archivo in o.listdir(rutacompleta):
        print('Imagenes: ',f+'/',archivo)
        ids.append(id)
        rostrosdata.append(cv.imread(rutacompleta+'/'+archivo,0))
        
    id=id+1
    tiempodefinaldelectura=time()
    tiempototallectura=tiempodefinaldelectura-tiempoinicial
    print('Tiempo Total: ',tiempototallectura)

entrenamientoRecognizer=cv.face.EigenFaceRecognizer_create()
print('Iniciando Entrenamiento...espere')
entrenamientoRecognizer.train(rostrosdata,np.array(ids))
tiempofinalentrenamiento=time()
tiempototalentrenamiento=tiempofinalentrenamiento-tiempototallectura
print('tiempo entrenamiento total: ',tiempototalentrenamiento)
entrenamientoRecognizer.write('EntrenamientoEigenFaceRecognizer.xml')
print('Entrenamiento Concluido')