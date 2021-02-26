import cv2
import os 
import time
import json

ts=time.gmtime()

c=os.path.dirname(__file__) # obtenção do diretório para gravação da timestamp
timestamp=c+"\\data.log"

detectorFace = cv2.CascadeClassifier("C:/Users/Jorge.silva/Downloads/ML/Curso_Udemy_ML/Reconhecimento_Facial_REV0201/haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.FisherFaceRecognizer_create()
reconhecedor.read("C:/Users/Jorge.silva/Downloads/ML/Curso_Udemy_ML/Reconhecimento_Facial_REV0201/classificadorFisher.yml")
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0) # referência de leitura da câmera

while (True):
    #print(time.strftime("%Y-%m-%d %H:%M:%S", ts))
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza,
                                                    scaleFactor=1.5,
                                                    minSize=(30,30))
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,255,0), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        
        nome = ""
        if id == 1:
            nome = 'Nay'
        elif id == 2:
            nome = 'Jorge'
        elif id == 3:
            nome = 'Rapha'
        else:
            nome = 'Fer'
            
        cv2.putText(imagem, nome, (x,y +(a+30)), font, 2, (0,255,0))
        #cv2.putText(imagem, str(confianca), (x,y + (a+50)), font, 1, (0,0,255))
        cv2.putText(imagem, time.strftime("%d-%m-%Y %H:%M:%S", ts), (x,y +(a+50)), font, 1, (0,255,0))        

                  
    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

def gravarDados():
    arquivo=open(timestamp,"a")
    arquivo.write(ts)
    arquivo.write("\n")
    arquivo.close()

camera.release()
cv2.destroyAllWindows()