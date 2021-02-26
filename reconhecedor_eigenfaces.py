import cv2

detectorFace = cv2.CascadeClassifier("C:/Users/Jorge.silva/Downloads/ML/Curso_Udemy_ML/Reconhecimento_Facial_REV0201/haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("C:/Users/Jorge.silva/Downloads/ML/Curso_Udemy_ML/Reconhecimento_Facial_REV0201/classificadorEigen.yml")
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0) #camera do notebook
camera = cv2.VideoCapture(1) #camera usb

while (True):
        conectado, imagem = camera.read()
        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        facesDetectadas = detectorFace.detectMultiScale(imagemCinza,
                                                       scaleFactor = 1.5, 
                                                       minSize =(30,30))
        for (x, y, l, a) in facesDetectadas:
            
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
            id, confianca = reconhecedor.predict(imagemFace)
            if id == 13374:
                nome = 'JORGE'
            elif id == 13449:
                nome = 'JANDERSON'
            elif id == 13452:
                nome = 'RAMON'
            elif id == 13210:
                nome = 'BARBA'
            elif id == 13346:
                nome = 'MURILO'
            elif id == 13075:
                nome = 'CHRISTYAM'
            elif id == 1:
                nome = 'NAY'
            else:
                nome = 'JOHN DOE'
            cv2.putText(imagem, nome, (x, y + (a+30)), font, 2, (0,255,0))
        cv2.imshow("Face", imagem)
        if cv2.waitKey(1) == ord('q' or 'Q'):
            break


camera.release()
cv2.destroyAllWindows()
    