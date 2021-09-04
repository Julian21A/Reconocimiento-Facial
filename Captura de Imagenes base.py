import cv2 as cv
import os as o
import imutils as im

modelo='Julian'
ruta1='C:/Users/hp/Documents/Python/Analisis Imagenes/Reconocimiento Facial/Data'
rutacompleta=ruta1+'/'+modelo
if not o.path.exists(rutacompleta):
    o.makedirs(rutacompleta)


camara = cv.VideoCapture(1)
ruido=cv.CascadeClassifier('C:/Users/hp/Documents/Python/Analisis Imagenes/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
id=0
while True:
    respuesta,captura= camara.read()
    if respuesta==False: break
    captura=im.resize(captura,width=640)

    grises=cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcaptura=captura.copy()
    cara=ruido.detectMultiScale(grises,1.3,7)
    
    for(x,y,e1,e2) in cara:
        cv.rectangle(captura,(x,y),(x+e1,y+e2),(255,5,50),2)
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado,(160,160),interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id),rostrocapturado)
        id=id+1
    
    cv.imshow("Resultado Rostro",captura)
    
    if id==501:
        break

camara.release()
cv.destroyAllWindows