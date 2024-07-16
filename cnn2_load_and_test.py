# -*- coding: utf-8 -*-
import lib.spectgr
import lib.spectgrList
import lib.tensores as tensores
from keras.utils import to_categorical
import pandas as pd
import os

# main(carpetaModelos,carpetaTensoresTest,carpetaResultados,numClases,numMFCCs,numChannels,numEspectros)
def main(tensoresTest,datosTest,carpetaModelos,nombreArchivoModelo,carpetaResultados,numClases):

    # Load cnn model
    from keras.models import load_model
    
    # Si el nombre del modelo no contiene la extensión .h5, se la añadimos
    if nombreArchivoModelo.find(".h5") == -1:
        nombreArchivoModelo=nombreArchivoModelo+".h5"

    nombreCompletoModelo=carpetaModelos+nombreArchivoModelo

   
    cnnModel = load_model(nombreCompletoModelo)

    # Si cnnModel es None, no se ha podido cargar el modelo
    if cnnModel==None:
        print("No se ha podido cargar el modelo: ",nombreCompletoModelo)
        exit()

    predictions = cnnModel.predict(tensoresTest)

    # Convierte predictions en dataframe
    predictions_df=pd.DataFrame(predictions)

    # Añade al dataframe resultados_df las columnas de predictions_df
    resultados_df=pd.concat([datosTest,predictions_df],axis=1)

    # Crea un dataframe vacío con las columnas: locutor, enunciado, oral, nasal
    # =pd.DataFrame(columns=["locutor","enunciado","oral","nasal"])

    data = {
    "locutor": [],
    "enunciado": [],
    "oral": [],
    "nasal": []}

    informeResultados_df = pd.DataFrame(data)

    enunciados=resultados_df['enunciado'].unique()
    locutores=resultados_df['locutor'].unique()
    for locutor in locutores:
        for enunciado in enunciados:
            row=[locutor,enunciado]
            # Selecciona las filas de resultados_df correspondientes al enunciado y locutor
            df=resultados_df.loc[(resultados_df['enunciado']==enunciado) & (resultados_df['locutor']==locutor)]
            # Calcula la media de las columnas 0 1 2
            listaClases=list(range(0,numClases))
            media=df[listaClases].mean()
            for clase in listaClases:
                row.append((media[clase]))
            
            # Si los valores de Row no son nan, añade la fila al dataframe resultados_df
            if not pd.isnull(row).any():
                informeResultados_df.loc[len(informeResultados_df)]=row

    # Guarda resultados_df en un csv
    fName=carpetaResultados+nombreArchivoModelo[:-3]+"_result.csv"  
    print("Resultados guardados en: ",fName)
    resultados_df.to_csv(fName,index=False)

    return 0

def iniciaCNN2(modelo,tensoresTest,datosTest,numEspectros):
    if numEspectros==21:
        tamanyoVentana=200
    elif numEspectros==26:
        tamanyoVentana=250
    elif numEspectros==31:
        tamanyoVentana=300
    else:
        print("Número de espectros no válido")
        exit()

    
    carpetaModelos="cnnModels/"
    carpetaTensoresTest="datosTest/tensores"+str(tamanyoVentana)+"/"
    carpetaResultados="cnnResults/"

    # print("Modelo: ",modelo)
    # print("carpetaModelos: ",carpetaModelos)
    # print("carpetaTensoresTest: ",carpetaTensoresTest)
    # print("carpetaResultados: ",carpetaResultados)
    numClases=2

    carpetaModelos=os.getcwd()+"/"+carpetaModelos
    nombreArchivoModelo="cnn_"+str(modelo)+".h5"
    nombreCompletoModelo=carpetaModelos+nombreArchivoModelo
    carpetaTensoresTest=os.getcwd()+"/"+carpetaTensoresTest
    carpetaResultados=os.getcwd()+"/"+carpetaResultados

    # # Si no existe, crea una carpeta "folder" en la carpeta base
    # if not os.path.exists(carpetaTensoresTest):
    #     print("La carpeta con los tensores Test existe: ",carpetaTensoresTest)
    #     exit()
    if not os.path.exists(nombreCompletoModelo):
        print("El modelo no existe: ",nombreCompletoModelo)
        exit()
    main(tensoresTest,datosTest,carpetaModelos,nombreArchivoModelo,carpetaResultados,numClases)

    print("Terminado el modelo:", nombreArchivoModelo)
