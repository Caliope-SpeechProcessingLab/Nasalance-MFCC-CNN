# -*- coding: utf-8 -*-
#!/usr/bin/env python
# Aplicación para comprobar diferentes parámetros de la CNN. 
# Se pueden modificcar los siguientesi parámetros (los valores indicados son de la configuración por defecto):
import sys
# Configuración base de la CNN
# 0. Dialecto
# 1. numCapas=2
# 2. activacionCapaSalida="softmax"
# 3. numFiltros=[16]
# 4. kernel de la primera capa (kernel11=True) o de todas las capas (kernel11=False)
#    kernels=[(2,1),(2,2),(3,1),(3,2),(4,1),(4,2),(1,3)] 
# 5. numModelos=5 Es el número de iteraciones para cada configuración
# 6. indicesMFCC=[0,1,2,3,4,5,6,7,8,9,10,11,12] (todos los MFCC) o [0] (solo el primero). Aunque se puede probar cualquier combinación,
#    el mffc0 debe estar en todo caso. 
# 7. nivelBorradoModelo=0/1/2 (nada/todo menos resultados/todo)

# Cada variable de las anteriores se puede configurar como una lista de valores, y se probarán todas las combinaciones

# Además se puede configurar:
# tamaño de ventana (200, 250 o 300) 
# numEspectros=26

from re import M
import lib.gestionAudioTrain as cnnGestionAudios
import cnn1_train_and_save as cnn1
import cnn2_load_and_test as cnn2
import cnn3_Report as cnn3
import lib.spectgrList
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import shutil
import platform
import config as cfg

# *********************
# Función principal 
# ********************* 

def todo(dialecto,kernel11,nombreInformeFinal):

#  recodifica el string kernel11 como booleano
#    if kernel11=="True":
#        kernel11=True
#    elif kernell11=="False":
#        kernel11=False
#    else:
#     print("Error: kernel11 debe ser True o False")
#     exit()
    
    
    listaNumCapas= [2]

    # PARÁMETRO 2. Función de activación de la capa de salida
    activacionCapaSalida="softmax"
    # activacionCapaSalida="sigmoid"

    # PARÁMETRO No configurable fácilmente: Tamaño de ventana 
    # (determina el tamaño de los espectrogramas MFCC)
    # También define la carpeta de los datos de entrenamiento y testeo

    tamanyoVentana=250

    if tamanyoVentana==200:
        numEspectros=21
    elif tamanyoVentana==250:
        numEspectros=26
    elif tamanyoVentana==300:
        numEspectros=31
    else:
        print("Tamaño de ventana no válido")
        print("El tamaño de ventana debe ser 200, 250 o 300")
        exit()

    # PARÁMETRO 3. Serie Número de filtros (solo para la primera capa, el resto son múltiplos de este valor)
    # Si se indican varios valores, se probarán todos los modelos
    serieNumFiltros=[16]

    # PARÁMETRO 4. kernel11 indica si en la primera capa se usa solo ese kernel. kernels indica los kernels se probaran 
    # ya sea en todas las capasa o en todas menos las primera
    kernels11=[True]
    kernels = []
    for i in range (1,9):
        for j in range (1,9):
            if i+j>2:
                kernels.append((i,j))
    #kernels=[(1,5),(1,4),(1,3),(3,2),(2,3),(3,3),(3,4),(4,3),(4,4),(5,5),(5,4),(5,3),(5,2),(2,5)]
    #kernels=[(6,1),(5,1),(4,1),(3,1),(1,6),(1,5),(1,4),(1,3),(3,2),(2,3),(3,3),(3,4),(4,3),(4,4),(5,5),(5,2),(2,5),(3,5),(5,3)]
    # PARÁMETRO 5. NumIteraciones (número de modelos que se van a entrenar y evaluar)
    numModelos=5

    # PARÁMETRO 6. Serie de índices de MFCC a usar
    # seriesIndicesMFCC=[[0],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6,7,8,9,10,11,12]]
    # seriesIndicesMFCC=[[0,1,2,3,4,5,6,7,8,9,10,11,12]]
    seriesIndicesMFCC=[[0,1,2,3,4,5,6,7,8,9,10,11,12,
                        13,14,15,16,17,18,19,20,21,22,23,24,25,
                        26,27,28,29,30,31,32,33,34,35,36,37,38]]
    seriesIndicesMFCC=[[0,1,2,3,4,5,6,7,8,9,10,11,12]]
    # seriesIndicesMFCC = cfg.SERIES_INDICES_MFCC
    # Estas dos configuración pueden servir para comparar qué ocurre cuando se usa solo la energía (MFCC0), como en nasalancia,
    #  o cuando se añade la info espectral

    # PARÁMETRO 7. Opciones de borrado de modelos y resultados
        # 0: no borrar nada;  
        # 1: se borra modelo; se conservan resultados (prob posteriores); 
        # 2: se borran modelos y resultados
    nivelBorradoModelo=0

    # PARÁMETRO 8. Carpeta de los datos de testeo
    # carpetaTesteoBase="datosTestEspectroTemporal/tensores"
    carpetaTesteoBase="datosTest/tensores_13MFCCs"

    # Función para borrar los archivos generados al crear un modelo
    def borraModelo(nuevoModelo,nivelBorrado):
        # Si el sistema operativo es Windows 10, se cambia el separador de carpetas
        if platform.system()=="Windows":
            ubicacionModelos="\\cnnModels\\cnn_"
            ubicacionResultados="\\cnnResults\\cnn_"
        else:
            ubicacionModelos="/cnnModels/cnn_"
            ubicacionResultados="/cnnResults/cnn_"

        if nivelBorrado > 0: 
            # Calcula la ruta completa del modelo
            rutaCompletaModelo=os.getcwd()+ubicacionModelos+nuevoModelo
            archivoModelo=rutaCompletaModelo+".h5"
            carpetaModelo=rutaCompletaModelo
            archivoResumen=os.getcwd()+ubicacionResultados+nuevoModelo+"_resumen.csv"
            # Comprueba si existe el archivoModelo
            if os.path.isfile(archivoModelo):
                os.remove(archivoModelo)
            else:
                print("No existe el archivo: ",archivoModelo)
            if os.path.isfile(archivoResumen):
                os.remove(archivoResumen)
            else:
                print("No existe el archivo: ",archivoResumen)
                exit()
            # Comprueba si existe la carpetaModelo
            if os.path.isdir(carpetaModelo):
                print("Borrando carpeta: ",carpetaModelo)
                # Borra la carpeta y sus contenidos
                shutil.rmtree(carpetaModelo)
            else:
                print("No existe la carpeta: ",carpetaModelo)
                exit()
        if nivelBorrado >1: # Se borran también los resultados
            archivoResultados=os.getcwd()+ubicacionResultados+nuevoModelo+"_result.csv"
            if os.path.isfile(archivoResultados):
                os.remove(archivoResultados)
            else:
                print("No existe el archivo: ",archivoResultados)
                exit()

    # Esto es por si solo se quiere probar train/test/report. Normalmente estará todo a 1
    train=0
    test=1
    report=1

    # Parámetro para usar una escala de 0 a 3
    escalas0_3=[True]

    # Crea dataFrame con una fila por cada locutor
    tensoresTest=[]
    datosTest=[]

    carpetaTensoresTest=carpetaTesteoBase+"/"
    carpetaTensoresTest=os.getcwd()+"/"+carpetaTensoresTest
    # print("carpetaTensoresTest: ",carpetaTensoresTest)
    tensoresTestOrig,datosTest=lib.spectgrList.read_FolderTestTensorData(carpetaTensoresTest,numEspectros, numMFCCs=13,numChannels=2)

    numTotalModelos=len(escalas0_3)*len(kernels)*len(seriesIndicesMFCC)*numModelos

    empiezapor=0
    modelosHechos=0
    # índices para usar todos los MFCC (solo una iteración)
    for escala0_3 in escalas0_3:
        for indicesMFCC in seriesIndicesMFCC:
            carpetaTrain="datosTrain/tensores"+dialecto
            carpetaTrain=os.getcwd()+"/"+carpetaTrain
            # Se define el nombre de los modelos, para aquellos parámetros que siempre tienen valores fijos. 

            # # Convierte la lista indicesMFCC en un string, separando los valores por un guión
            indicesMFCCStr="-".join(str(x) for x in indicesMFCC)

            # Seleccionar los MFCC indicados
            print("Indices MFCC: ",indicesMFCC)
            tensoresTest = tf.gather(tensoresTestOrig, indicesMFCC, axis=1)

            # Verificar la forma del tensor seleccionado
            print("Forma del tensor seleccionado:", tensoresTest.shape)

            for numCapas in listaNumCapas: 
                for numFilters in serieNumFiltros:
                    for kernel in kernels:
                        k1=str(kernel[0])
                        k2=str(kernel[1])
                        kernel=(int(k1),int(k2))
                        # lista_rSpearmanNLCET2T3T5T6=[]
                        lista_rSpearmanCNNT2T3T5T6=[]
                        # for modelo in range(numModelos):
                        for modelo in range(numModelos):
                            modelosHechos+=1
                            # print("*****************************")
                            print("Modelo: ",modelosHechos," de ",numTotalModelos)

                            if kernel11==True:
                                nuevoModelo=dialecto+"_k11"+"_k"+k1+k2+"_c"+str(numCapas)+"_f"+str(numFilters)+"_n"+str(modelo) 
                            else:
                                nuevoModelo=dialecto+"_k"+k1+k2+"_k"+k1+k2+"_c"+str(numCapas)+"_f"+str(numFilters)+"_n"+str(modelo) 
                            # print("Modelo: ",nuevoModelo)
                            if train==1:
                                cnn1.iniciaCNN1(nuevoModelo,carpetaTrain,kernel,numCapas,numEspectros,activacionCapaSalida,numFilters,indicesMFCC,kernel11, num_crossval=modelo) 
                            # if finetunning==1:
                            #     cnn1b.iniciaCNN1b(nuevoModelo)
                            if test==1:
                                print("Voy a empezar el testeo: cnn2")
                                cnn2.iniciaCNN2(nuevoModelo,tensoresTest=tensoresTest,datosTest=datosTest,numEspectros=numEspectros)
                            if report==1 and modelosHechos>=empiezapor:
                                n=cnn3.iniciaCNN3(nuevoModelo,nombreInformeFinal,escala0_3)
                                # lista_rSpearmanCNNT2T3T5T6.append(speCNNT2T3T5T6r)

                            # Una vez hecho el informe borramos el modelo para no cargar el disco duro
                            if train==1 and nivelBorradoModelo>0:
                                borraModelo(nuevoModelo,nivelBorradoModelo)

                        # # media_rSpearmanNLCET2T3T5T6=round(sum(lista_rSpearmanNLCET2T3T5T6)/len(lista_rSpearmanNLCET2T3T5T6),3)            # Redondea a tres decimales la variable media_rSpearmanT2T5T6
                        # media_rSpearmanCNNT2T3T5T6=round(sum(lista_rSpearmanCNNT2T3T5T6)/len(lista_rSpearmanCNNT2T3T5T6),3)

                        # # print("Media rSpearmanNCLET2T3T5T6: ",media_rSpearmanNLCET2T3T5T6)
                        # print("Media rSpearmanCNNT2T3T5T6: ",media_rSpearmanCNNT2T3T5T6)

    return 0


# Este script permite ejecutar todo con tres parámetros
# nombreInformeFinal="spearman_Asica2vsAsica3vsAsica4vsAsica5_STG_K11True.csv"
# dialecto="Asica2ESP"
# kernel11=True
# Sirve para dividir un trabajo en varios scripts para Picasso, y que cada uno haga una parte
# todo(nombreInformeFinal,dialecto,kernel11)

dialecto=sys.argv[1]
paramKernel11=sys.argv[2]
nombreInforme=sys.argv[3]

if paramKernel11=="True":
    kernel11=True
elif paramKernel11=="False":
    kernel11=False
else:
    print("Error: el parámetro debería ser kernel11 True o False")
    exit()

todo(dialecto,kernel11,nombreInforme)

