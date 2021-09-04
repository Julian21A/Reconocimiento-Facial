import cv2 as cv
import os as o

dataruta='C:/Users/hp/Documents/Python/Analisis Imagenes/Reconocimiento Facial/Data'
listadata=o.listdir(dataruta)
entrenamientoRecognizer=cv.face.EigenFaceRecognizer_create()
entrenamientoRecognizer.read('C:/Users/hp/Documents/Python/Analisis Imagenes/EntrenamientoEigenFaceRecognizer.xml')
ruidos=cv.CascadeClassifier('C:/Users/hp/Documents/Python/Analisis Imagenes/Reconocimiento Facial/haarcascade_frontalface_default.xml')
camara=cv.VideoCapture(1)

while True:
    respuesta,captura=camara.read()
    if respuesta==False: break
    
    grises=cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcaptura=grises.copy()
    cara=ruidos.detectMultiScale(grises,1.3,5)
    for(x,y,e1,e2) in cara:
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado,(160,160),interpolation=cv.INTER_CUBIC)
        resultado=entrenamientoRecognizer.predict(rostrocapturado)
        cv.putText(captura,'{}'.format(resultado),(x,y-5),1,1.3,(255,0,100),1,cv.LINE_AA)
        if resultado[1]<9000:
            cv.putText(captura,'{}'.format(listadata[resultado[0]]),(x,y-20),2,0.7,(255,0,100),1,cv.LINE_AA)           
            cv.rectangle(captura,(x,y),(x+e1,y+e2),(100,100,100),2)
        else:
            cv.putText(captura,'No encontrado' ,(x,y-20),2,1.3,(255,0,100),1,cv.LINE_AA)
            cv.rectangle(captura,(x,y),(x+e1,y+e2),(100,100,100),2)    

    cv.imshow("Resultados",captura)
    if cv.waitKey(1)==ord('s'): break
camara.release()
cv.destroyAllWindows()