import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('C:/Users/Jorge.silva/Downloads/ML/Curso_Udemy_ML/Reconhecimento_Facial_REV0201/fotos2',f) for f in os.listdir('C:/Users/Jorge.silva/Downloads/ML/Curso_Udemy_ML/Reconhecimento_Facial_REV0201/fotos2')]
    #print(caminhos)
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem) [-1].split('.')[1])
        #print(id)
        ids.append(id)
        faces.append(imagemFace)
        #cv2.imshow("Face", imagemFace)
        #cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = getImagemComId()
#print(faces)

print("Treinando...")
eigenface.train(faces, ids)
eigenface.write('C:/Users/Jorge.silva/Downloads/ML/Curso_Udemy_ML/Reconhecimento_Facial_REV0201/classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('C:/Users/Jorge.silva/Downloads/ML/Curso_Udemy_ML/Reconhecimento_Facial_REV0201/classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('C:/Users/Jorge.silva/Downloads/ML/Curso_Udemy_ML/Reconhecimento_Facial_REV0201/classificadorLBPH.yml')
print("Treinamento realizado")

#getImagemComId()