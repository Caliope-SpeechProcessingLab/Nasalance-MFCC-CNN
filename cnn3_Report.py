# -*- coding: utf-8 -*-
# Este archivo se encarga de analizar los resultados obtenidos por cnn2_load_and_test.py

from math import e
from re import T
from unittest import result
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

soloPacientes=False
excluirHypoNasales=False

def listaACadena(lista):
    cadena=""
    for elemento in lista:
        cadena+=str(elemento)
        if elemento!=lista[-1]:
            cadena+="\t"
    return cadena

def informeXLocutor(informeGeneralResumen,resultadosModeloActual,resultPerNlace,fresumen,modelo,locutores,escala0_3):
    # print("fresumen:",informeGeneralResumen)
    # print("modelo:",modelo)
    
    hipoNasalales=excluir=["fin97508a","fin97508b"]
    hipoNasalales=excluir=[]
    
    # Abre el archivo de resultados resultsFile
    df_resultadosModeloActual=pd.read_csv(resultadosModeloActual)

    if locutores=="all":      
        # Obtiene la lista de locutores y la ordena alfabéticamente
        locutoresCNN=df_resultadosModeloActual['locutor'].unique()
        locutoresCNN.sort()
        # print("locutoresCNN_all:",locutoresCNN)
    else:
        # Selecciona las filas de resultPerNlace en las que la columna Grupo valga 1
        resultPerNlace=resultPerNlace[resultPerNlace['Grupo']==locutores]
        # Obtiene la lista de locutores y la ordena alfabéticamente
        locutoresCNN=resultPerNlace['locutor'].unique()
        locutoresCNN.sort()
    
    # Obtén la lista de locutores de resultPerNlace
    locutoresPerc=resultPerNlace['locutor'].unique()
    # Por algún motivo el último element es nan. Lo elimino, si es así:
    ultimo: str=locutoresPerc[-1]
    # print("LocutoresPerc:",locutoresPerc)
    # print("ultimo:",ultimo)
    # exit()
    # if np.isnan(ultimo):
    #     locutoresPerc=locutoresPerc[:-1]
    locutoresPerc.sort()

    # Comprueba si las listas locutresCNN y locutoresPerc son iguales
    locutoresIguales=True
    if len(locutoresCNN)==len(locutoresPerc):
        for i in range(len(locutoresCNN)):
            if locutoresCNN[i]!=locutoresPerc[i]:
                locutoresIguales=False
                break 
    else:
        locutoresIguales=False

    if not locutoresIguales:
        print("Error: la lista de locutores evaluados por la CNN y la lista del informe de evaluación perceptual no coinciden")
        print("Se aborta el programa")
        print("locutoresPerc:",locutoresPerc)
        print("locutoresCNN:",locutoresCNN)
        exit()

    medidas=['T20','T5','T6','T2T5','T2T6','T2T5T6','T12','T14','T17','T2','T32']
    medidas=['T2','T5','T6','T2T5T6']

    T2=["T2pa", "T2pi", "T2ta", "T2ti", "T2ka", "T2ki"]
    T5=["T5boca", "T5pie", "T5llave", "T5dedo", "T5gafas", "T5silla", "T5cuchara", "T5sol", "T5casa", "T5pez", "T5jaula", "T5zapato"]
    T5=["T5boca", "T5pie", "T5llave", "T5dedo", "T5gafas", "T5silla", "T5sol", "T5casa", "T5pez"]
    T6=["T6AlGato", "T6ADavid", "T6UyAhiHayAlgo", "T6SiLlueveLeLlevoLaLlave", "T6SusiSaleSola", "T6FaliFueFeria", "T6LosZapatosDeCecilia", "T6LaJirafaJesus", "T6TodaLaTazaDeTe", "T513_PapaPuedePelarAPili", "T6QuiqueCoge"]
    # T2T5=T2+T5
    # T2T6=T2+T6
    T2T5T6=T2+T5+T6
    # T12=["T2pa", "T2pi", "T2ta", "T2ti", "T2ka", "T2ki", "T5pie", "T5dedo", "T5pez", "T5jaula", "T6AlGato", "T6ADavid"]
    # T14=T12+["T406_llave", "T6SusiSaleSola"]
    # T17=T14+["T5sol", "T6PapaPuedePelar", "T6QuiqueCoge"]
    # T20=T17+["T5cuchara", "T5silla", "T6LaJirafaJesus"]
    # T32=T20+["T3f", "T3s", "T3a","T5boca", "T5gafas", "T5casa", "T5zapatos", "T6UyAhiHayAlgo", "T6SiLlueveLeLlevoLaLlave", "T6FaliFueFeria", "T6LosZapatosDeCecilia", "T6TodaLaTazaDeTe"]
    cnn_T2=[]
    cnn_T5=[]
    cnn_T6=[]
    # cnn_T2T5=[]
    # cnn_T2T6=[]
    cnn_T2T5T6=[]
    # cnn_T12=[]
    # cnn_T14=[]
    # cnn_T17=[]
    # cnn_T20=[]
    # cnn_T32=[]

    # Muestra valores únicos de la columna enunciado
    # print("Valores únicos de la columna enunciado:",df_resultadosModeloActual['enunciado'].unique())
    # exit()
    # Para cada locutor de resultsCNNRaw
    for locutor in locutoresCNN:
        # print("locutor:",locutor)
        # Selecciono las filas de ese locutor en las que el enunciado está en T2Orales
        dfLocutor=df_resultadosModeloActual[df_resultadosModeloActual['locutor']==locutor]
        # Crea un DF para cada serie de enunciados
        # dfLocutorT20=dfLocutor[dfLocutor['enunciado'].isin(T20)]
        dfLocutorT2=dfLocutor[dfLocutor['enunciado'].isin(T2)]
        dfLocutorT5=dfLocutor[dfLocutor['enunciado'].isin(T5)]
        dfLocutorT6=dfLocutor[dfLocutor['enunciado'].isin(T6)]
        # dfLocutorT2T5=dfLocutor[dfLocutor['enunciado'].isin(T2T5)]
        # dfLocutorT2T6=dfLocutor[dfLocutor['enunciado'].isin(T2T6)]
        dfLocutorT2T5T6=dfLocutor[dfLocutor['enunciado'].isin(T2T5T6)]
        # dfLocutorT12=dfLocutor[dfLocutor['enunciado'].isin(T12)]
        # dfLocutorT14=dfLocutor[dfLocutor['enunciado'].isin(T14)]
        # dfLocutorT17=dfLocutor[dfLocutor['enunciado'].isin(T17)]
        # dfLocutorT32=dfLocutor[dfLocutor['enunciado'].isin(T32)]

        
        # Calculo la media de la columna 1 en cada caso
        # mediaNasalT20=dfLocutorT20['1'].mean()
        mediaNasalT2=dfLocutorT2['1'].mean()
        mediaNasalT5=dfLocutorT5['1'].mean()
        mediaNasalT6=dfLocutorT6['1'].mean()
        # mediaNasalT2T5=dfLocutorT2T5['1'].mean()
        # mediaNasalT2T6=dfLocutorT2T6['1'].mean()
        mediaNasalT2T5T6=dfLocutorT2T5T6['1'].mean()
        # mediaNasalT12=dfLocutorT12['1'].mean()
        # mediaNasalT14=dfLocutorT14['1'].mean()
        # mediaNasalT17=dfLocutorT17['1'].mean()
        # mediaNasalT32=dfLocutorT32['1'].mean()

        # Añado la media a la lista CNN correspondiente
        # cnn_T20.append(round(mediaNasalT20,2))
        cnn_T2.append(round(mediaNasalT2,2))
        cnn_T5.append(round(mediaNasalT5,2))
        cnn_T6.append(round(mediaNasalT6,2))
        # cnn_T2T5.append(round(mediaNasalT2T5,2))
        # cnn_T2T6.append(round(mediaNasalT2T6,2))
        cnn_T2T5T6.append(round(mediaNasalT2T5T6,2))
        # cnn_T12.append(round(mediaNasalT12,2))
        # cnn_T14.append(round(mediaNasalT14,2))
        # cnn_T17.append(round(mediaNasalT17,2))
        # cnn_T32.append(round(mediaNasalT32,2))


    # listasCNN=[cnn_T20,cnn_T5,cnn_T6,cnn_T2T5,cnn_T2T6,cnn_T2T5T6,cnn_T12,cnn_T14,cnn_T17,cnn_T2,cnn_T32]
    listasCNN=[cnn_T2,cnn_T5,cnn_T6,cnn_T2T5T6]
    # Normaliza las listas cnn_
    if escala0_3:
            for listaCNN in listasCNN:
            # Crea una copia de la lista y la ordena de menor a mayor
                nuevaLista=listaCNN.copy()
                nuevaLista.sort()
                max0=nuevaLista[26]
                max1=nuevaLista[33]
                max2=nuevaLista[44]
                for i in range(len(listaCNN)):
                    if listaCNN[i]<max0:
                        listaCNN[i]=0
                    elif listaCNN[i]<max1:
                        listaCNN[i]=1
                    elif listaCNN[i]<max2:
                        listaCNN[i]=2
                    else:
                        listaCNN[i]=3
    # print("lista CNN 1:",listasCNN[0])
    # exit()
    resultPerNlace.loc[:,'CNN_T2']=listasCNN[0]
    resultPerNlace.loc[:,'CNN_T5']=listasCNN[1]
    resultPerNlace.loc[:,'CNN_T6']=listasCNN[2]
    # resultPerNlace.loc[:,'CNN_T2T5']=listasCNN[3]
    # resultPerNlace.loc[:,'CNN_T2T6']=listasCNN[4]
    resultPerNlace.loc[:,'CNN_T2T5T6']=listasCNN[3]
    # resultPerNlace.loc[:,'CNN_T12']=listasCNN[6]
    # resultPerNlace.loc[:,'CNN_T14']=listasCNN[7]
    # resultPerNlace.loc[:,'CNN_T17']=listasCNN[8]
    # resultPerNlace.loc[:,'CNN_T20']=listasCNN[9]
    # resultPerNlace.loc[:,'CNN_T32']=listasCNN[10]

    if soloPacientes:
        # Selecciona las filas de resultPerNlace en las que la columna locutor no empieza por 'nas'
        resultPerNlace=resultPerNlace[~resultPerNlace['locutor'].str.startswith('nas')]

    if excluirHypoNasales:
        # Selecciona las filas de resultPerNlace en las que la columna locutor no está en la lista hipoNasalales
        resultPerNlace=resultPerNlace[~resultPerNlace['locutor'].isin(hipoNasalales)]

    # # Comprueba si la distribución de perc_T2T5T6 es normal
    # statPerc, p_valor = shapiro(resultPerNlace['Perc_T2T5T6']) 
    # print("Normaltest Perc: ",statPerc)
    # # Comprueba si la distribución de cnn_T2T5T6 es normal
    # statCNN=shapiro(resultPerNlace['CNN_T2T5T6'])
    # print("Normaltest CNN: ",statCNN)

    # # Comprueba si la distrinución de nlce_T2T5T6 es normal
    # statNlce=shapiro(resultPerNlace['Nlce_T2T5T6'])
    # print("Normaltest NLCE: ",nNlce)

    # exit()

    serieMedidas=listaACadena(medidas)
    cabeceraInforme="Dialecto	kC1	kC2	It	"+str(serieMedidas)+"\n"
    listaR=[]
    for medida in medidas:
        colPerc='Perc_'+medida
        colCNN='CNN_'+medida
        # colNlce='Nlce_'+medida

        # Calcula el p-valor
        rho, pval = spearmanr(resultPerNlace[colPerc], resultPerNlace[colCNN])
        spearmanCorr = round(rho,3)
        sig=pval
        listaR.append(spearmanCorr)

        # muestra los resultados por pantalla
        # print(medida,": ",spearmanCorr)

    # Convierte listaR en una cadena
    listaCadenaR=listaACadena(listaR)

    cadenaCNN="CNN"+modelo+"\t"+str(listaCadenaR)+"\n"
    
    # Si cadena CNN empieza por "CNNadultTodosESP_w250_softmax_", reemplaza por "ESP"
    if cadenaCNN.startswith("CNNadultTodosESP_w250_softmax_"):
        cadenaCNN=cadenaCNN.replace("CNNadultTodosESP_w250_softmax_","ESP\tDial")
    elif cadenaCNN.startswith("CNNadultTodosSJS_w250_softmax_"):
        cadenaCNN=cadenaCNN.replace("CNNadultTodosSJS_w250_softmax_","SJS\tDial")
    elif cadenaCNN.startswith("CNNadultTodosSTG_w250_softmax_"):
        cadenaCNN=cadenaCNN.replace("CNNadultTodosSTG_w250_softmax_","STG\tDial")
    elif cadenaCNN.startswith("CNNadultTodosSJSSTG_w250_softmax_"):
        cadenaCNN=cadenaCNN.replace("CNNadultTodosSJSSTG_w250_softmax_","SJSSTG\tDial")
    elif cadenaCNN.startswith("CNNadultTodosESPSJSSTG_w250_softmax_"):
        cadenaCNN=cadenaCNN.replace("CNNadultTodosESPSJSSTG_w250_softmax_","ESPSJSSTG\tDial")
    else:
        print("No sé por qué dialecto empieza")
        print("cadenaCNN:",cadenaCNN)
            
    # Sustituye "_c2_f16_" por "\t"
    cadenaCNN=cadenaCNN.replace("c2_f16_","\t")
    cadenaCNN=cadenaCNN.replace("k11_","k11\t")
    cadenaCNN=cadenaCNN.replace("_","\t")
    cadenaCNN=cadenaCNN.replace("\t\t","\t")
    # Si cadenaCNN empieza contiene "dial\tk11" reemplaza por "\tk11"
    if cadenaCNN.startswith("Dial\tk11"):
        cadenaCNN=cadenaCNN.replace("Dial\tk11","\tk11")
    else:        
        cadenaCNN=cadenaCNN.replace("Dial","")

    if os.path.exists(informeGeneralResumen): # type: ignore
        # print("Voy a escribir en el archivo Resumen: ",informeGeneralResumen)
        informeGeneralResumen=open(informeGeneralResumen,"a")
        informeGeneralResumen.write(cadenaCNN)
        # informeGeneralResumen.write(cadenaNLCE)
    # Si ya existe, lo abre y añade el resultado
    else:
        # print("Voy a crear el archivo fresumen: ",informeGeneralResumen)
        informeGeneralResumen=open(informeGeneralResumen,"w") # type: ignore
        informeGeneralResumen.write(cabeceraInforme+"\n")
        informeGeneralResumen.write(cadenaCNN)
        # informeGeneralResumen.write(cadenaNLCE)
    # Añade final de línea al archivo fresumen
    # informeGeneralResumen.write("\n")
    # Cierra el archivo fresumen
    informeGeneralResumen.close()

    # Guarda resultPerNlace en un csv
    fName=resultadosModeloActual[:-11]+"_resumen.csv"
    resultPerNlace.to_csv(fName,index=False,sep='\t')

    return 0

def clasificaLocutores(locutores):

    locutoresControl=[]
    locutoresPacientes=[]

    for locutor in locutores:
        if locutor.startswith("fin") or locutor.startswith("bcn"):
            locutoresControl.append(locutor)
        else:
            locutoresPacientes.append(locutor)

    # Muestra los locutores de control y los pacientes
    # print("Locutores de control: ",locutoresControl)
    # print("Locutores pacientes: ",locutoresPacientes)

    return locutoresControl, locutoresPacientes

def informeGrupal(resultsFile,fresumen):
    # Abre el archivo de resultados resultsFile
    df=pd.read_csv(resultsFile)

    listalocutores=df['locutor'].unique()
    locutoresControl, locutoresPacientes=clasificaLocutores(listalocutores)

    # Selecciona los datos de los controles y los pacientes
    dfControles=df[df['locutor'].isin(locutoresControl)]
    dfPacientes=df[df['locutor'].isin(locutoresPacientes)]

    # Bucle enunciados  
    enunciados=df['enunciado'].unique()
    for enunciado in enunciados:
        # Selecciona los datos de los controles y los pacientes
        dfOralControles=dfControles[dfControles['enunciado']==enunciado]
        dfOralPacientes=dfPacientes[dfPacientes['enunciado']==enunciado]

    # Haz una prueba t entre la columna '1' de dfOralControles y la columna '1' de dfOralPacientes
    # Realizar una prueba t independiente
        t_statistic, p_value = ttest_ind(dfOralControles['0'], dfOralPacientes['0'])

        # Pasa t_statistic y p_value a string
        t_statistic=str(round(t_statistic,3))
        p_value=str(round(p_value,10))

        # Resultado para el informe
        cadena="Enunciado: "+enunciado+". TTest: "+t_statistic+" pValue: "+p_value+"\n"
    
        # print("Resultado: ",cadena)

    # # Si existe el archivo fresumen, lo cañade el resultado; si no, lo crea de nuevo y añade el resultado
    #     if os.path.exists(fresumen):
    #         print("Voy a escribir el archivo fresumen: ",fresumen)
    #         fresumen=open(fresumen,"a")
    #         fresumen.write(cadena)
    #     # Si no existe, lo crea y añade el resultado
    #     else:
    #         print("Voy a crear el archivo fresumen: ",fresumen)
    #         fresumen=open(fresumen,"w")
    #         fresumen.write(cadena)
    # # Cierra el archivo fresumen
    # fresumen.close()
    exit()
    return 0

def main(nombreModelo,informeResultados,carpetaResultados,tipoInforme,nombreInformeFinal,escala0_3):
    

    # rSpearmanT2T5T6,rSpearmanT2T3T5T6=0,0

    baseDir=os.getcwd()
    if escala0_3:
        resultPercNlce=baseDir+"/"+carpetaResultados+"res_NlacePercEscala.txt"
    else:
        resultPercNlce=baseDir+"/"+carpetaResultados+"res_NlacePerc.txt"
    fresumen=informeResultados[:-4]+"_resumen.txt"
    informeGeneralResumen=baseDir+"/"+carpetaResultados+nombreInformeFinal+".txt"

    if not os.path.exists(resultPercNlce):
        print("El informe con los datos perceptuales y de nasalancia no existe: ",resultPercNlce)
        exit()
    else:
        resultPercNlce_df=pd.read_csv(resultPercNlce, delimiter='\t')
        # print("Informe resultPerNlce:",resultPercNlce_df.head())
        # exit()

    if tipoInforme=="Grupal":
        print("Informe grupal. PENDIENTE DE PROGRAMAR CÓDIGO")
        # print("fresumen:",fresumen)
        informeGrupal(informeResultados,fresumen)
    elif tipoInforme=="xLocutor":
        # print("Informe por locutor: ",informeResultados)
        speCNNT2T3T5T6r=informeXLocutor(informeGeneralResumen,informeResultados,resultPercNlce_df,fresumen,nombreModelo,"all",escala0_3)
        return speCNNT2T3T5T6r
    elif tipoInforme=="xEnunciado":
        print("Pendiente de programar: INFORME POR ENUNCIADO")
        # informeXEnunciado(informeResultados)    
    return -1

def iniciaCNN3(modelo,nombreInformeFinal,escala0_3):
    grupoLocutores="all"
    tipoInforme="xEnunciado"
    tipoInforme="Grupal"
    tipoInforme="xLocutor"

    carpetaResultados="cnnResults/"
    carpetaBase=os.getcwd()+"/"
    if grupoLocutores=="all":
        informeResultados=carpetaBase+carpetaResultados+"cnn_"+str(modelo)+"_result.csv"
    else:
        informeResultados=carpetaBase+carpetaResultados+"cnn_"+str(modelo)+"_result_"+grupoLocutores+".csv"

    # Si el informe no existe salimos
    if not os.path.exists(informeResultados):
        print("El informe no existe: ",informeResultados)
        return 0

    speCNNT2T3T5T6r=main(modelo,informeResultados,carpetaResultados,tipoInforme,nombreInformeFinal,escala0_3)

    return speCNNT2T3T5T6r
