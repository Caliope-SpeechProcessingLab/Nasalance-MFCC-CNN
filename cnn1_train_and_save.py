# -*- coding: utf-8 -*-
from sklearn import base
import lib.spectgr
import lib.spectgrList
import os
import numpy as np
from keras.utils import to_categorical
import lib.tensores as tensores
import tensorflow as tf
import keras.losses as losses
import keras.optimizers as optimizers
from sklearn.model_selection import KFold

# Para evitar el warning de que no se podrá cargar el modelo en el futuro. Es falso, sí que se puede cargar. 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def categorize_nasality(y_nasalityData,threshold):
    y_nasalityCat=np.empty(0)
    for dato in y_nasalityData:
        found=False
        classNasality=0
        while not found and classNasality<len(threshold):
            if dato >= threshold[classNasality][0] and dato <= threshold[classNasality][1]:
                y_nasalityCat=np.append(y_nasalityCat,classNasality)
                found=True
            classNasality+=1
        if found==False:
            y_nasalityCat=np.append(y_nasalityCat,-1)

    # Count how many spectrograms are in each class

    numClasses=len(threshold)
    print("Número de espectrogramas por clase: ")
    # Cunenta el número de elementos en cada clase
    for i in range(numClasses):
        print("Clase ",i,": ",np.count_nonzero(y_nasalityCat == i))
    return y_nasalityCat



def get_fold(df, tensor_data, fold_idx, num_folds=5):
  '''Realiza la validación cruzada dividiendo los datos en train y validación'''
  # Ensure locutor is the index to group by
  grouped = df.groupby('locutor')

  # Set the random seed for numpy for reproducibility
  np.random.seed(42)
  
  # Split groups into folds
  unique_locutors = list(grouped.groups.keys())
  np.random.shuffle(unique_locutors)  # Shuffle to ensure random splitting
  kf = KFold(n_splits=num_folds, random_state=42, shuffle=True)
  
  fold_locutors = list(kf.split(unique_locutors))
  
  # Get train and validation locutors
  train_locutors = [unique_locutors[i] for i in fold_locutors[fold_idx][0]]
  val_locutors = [unique_locutors[i] for i in fold_locutors[fold_idx][1]]
  print(f'Validation locutores en k-fold {fold_idx}: {val_locutors}' )
  # Get train and validation indices
  train_indices = df[df['locutor'].isin(train_locutors)].index
  val_indices = df[df['locutor'].isin(val_locutors)].index
  
  # Split data
  x_train = tensor_data[train_indices]
  y_train = df.iloc[train_indices]
  x_val = tensor_data[val_indices]
  y_val = df.iloc[val_indices]
  
  return x_train, y_train, x_val, y_val


# El formato de las imágenes es (numMFCCs=13,numEspectros=21/26,numChannels=2)
# Si hay un canal, el número finalde MFCCs será el doble (imágenes en escala de grises de 2*numMFCCs x numEspectros)
# Tensores (26,numEspectros,1)
# Si hay dos canales, el número final de MFCC será 13 (imágenes en color de 13 x numEspectros)
# Tensores (13,numEspectros,2)

def main(rutaCompletaModelo,tensorDataFolder,numMFCCs,numEspectros,numChannels,num_classes,kernel1x1,kernel,numCapas,activacionCapaSalida,numFilters,indicesMFCC, num_crossval=0):

    numMFCCs=len(indicesMFCC)

    print("*******************************************")
    print( "Inicio del script: cnn1_train_and_save.py")
    print("TrainFolderData: ",tensorDataFolder)
    print("Modelo: ",rutaCompletaModelo)
    print("kernel: ",kernel)
    print("numCapas: ",numCapas)
    print("numEspectros: ",numEspectros)
    print("Activation: ",activacionCapaSalida)

    modelName=rutaCompletaModelo

    # # Si hay un solo canal doblamos el número de MFCCs
    # if numChannels==1:
    #     numMFCCs=numMFCCs*2
    # else:
    #     numMFCCs=numMFCCs

    #  Lee datos de la carpeta indicada en Folder
    x_allData,y_AllData=lib.spectgrList.read_FolderTrainTensorData(tensorDataFolder)
    # x_allData is a list of tensors
    # y_AllData is a pandas dataframe
    
    print("Shape de x_allData: ", x_allData.shape)
    print("Shape de y_AllData: ", y_AllData.shape)

    # This shows how many nasality classes are established, and the maximum and minimum values of each class
    # threshold=[[0.0,0.05],[0.1,0.6],[0.6,1.0]]
    threshold=[[0.0,0.01],[0.3,1.0]]


    x_train, y_train, x_val, y_val = get_fold(y_AllData, x_allData, num_crossval)
    # x_train is a list of tensors
    # y_train is a pandas dataframe
    # x_val is a list of tensors
    # y_val is a pandas dataframe

    # Lee la columna ratioNasal del dataFrame y_AllData y la convierte en un array
    nasalityData_train=y_train['ratioNasal'].values
    print('Train')
    y_nasalityCat_train=categorize_nasality(nasalityData_train,threshold)
    print('Validacion')
    nasalityData_val=y_val['ratioNasal'].values
    y_nasalityCat_val=categorize_nasality(nasalityData_val,threshold)


    train_X_train=x_train
    train_X_val=x_val
    train_Y_train=y_nasalityCat_train
    train_Y_val=y_nasalityCat_val
    print("Shape de train_Y tras y_nasalityCat: ", train_Y_train.shape, train_Y_val.shape)

    # Convierto X_train a np.array
    train_X_train=np.array(train_X_train)
    train_X_val=np.array(train_X_val)

    train_X_train,train_Y_train=tensores.balancear(train_X_train,train_Y_train)
    train_X_val,train_Y_val=tensores.balancear(train_X_val,train_Y_val)

    print("Shape de train_X_train y val tras balancear: ", train_X_train.shape, train_X_val.shape)
    print("Shape de train_Y_train y val tras balancear: ", train_Y_train.shape, train_Y_val.shape)

     # Cambio la estructua de los datos de numEspectrograma x numChannels numMFCCs x numEspectros 
     # a numEspectrograma x numMFCCs x numEspectros x numChannels
    train_X_train = train_X_train.reshape(-1, numMFCCs, numEspectros,numChannels)
    train_X_val = train_X_val.reshape(-1, numMFCCs, numEspectros,numChannels)

    # Muestra la estructura de los datos
    print("Shape de train_X tras cambiar la estructura: ", train_X_train.shape, train_X_val.shape)
    print("Shape de train_Y: ", train_Y_train.shape, train_Y_val.shape)

    train_X_train=tensores.normalizaTensores(train_X_train)
    train_X_val=tensores.normalizaTensores(train_X_val)

    print("Shape de train_X: ", train_X_train.shape, train_X_val.shape)
    print("Shape de train_Y antes de one_hot: ", train_Y_train.shape, train_Y_val.shape)

    # Change the labels from categorical to one-hot encoding
    train_Y_one_hot = to_categorical(train_Y_train)
    valid_Y_one_hot = to_categorical(train_Y_val)

    print("Shape de train_X: ", train_X_train.shape, train_X_val.shape)
    print("Shape de train_Y_one_hot: ", train_Y_one_hot.shape, valid_Y_one_hot.shape)

    from sklearn.model_selection import train_test_split

    # train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
    train_X, valid_X, train_label, valid_label = train_X_train, train_X_val, train_Y_one_hot, valid_Y_one_hot

# Nos quedamos con los MFCCs especificados en indicesMFCC
    # train_X = tf.gather(train_X, indicesMFCC, axis=1)
    # valid_X = tf.gather(valid_X, indicesMFCC, axis=1)

    print("Shape de train_X: ", train_X.shape)
    print("Shape de train_label: ", train_label.shape)
    print("Shape de valid_X: ", valid_X.shape)
    print("Shape de valid_label: ", valid_label.shape)

    # Creamos la CNN

    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import LeakyReLU
    # from keras.initializers import glorot_uniform
    from keras.layers import BatchNormalization

    alpha=.1
    
    batch_size = 64
    epochs = 60

    AsicaCNNModel = Sequential()
    # Si el kernel es 1x1, se usa un filtro de 1x1 en la primera capa
    if kernel1x1==True:
        AsicaCNNModel.add(Conv2D(numFilters, kernel_size=(1,1),activation='linear',input_shape=(numMFCCs,numEspectros,numChannels),padding='same'))
    else:
        AsicaCNNModel.add(Conv2D(numFilters, kernel_size=(kernel),activation='linear',input_shape=(numMFCCs,numEspectros,numChannels),padding='same'))
    AsicaCNNModel.add(BatchNormalization())
    AsicaCNNModel.add(LeakyReLU(alpha=alpha))
    AsicaCNNModel.add(MaxPooling2D((2, 2),padding='same'))
    if numCapas>=2:
        AsicaCNNModel.add(Conv2D(numFilters*2, (kernel), activation='linear',padding='same'))
        AsicaCNNModel.add(BatchNormalization())
        AsicaCNNModel.add(LeakyReLU(alpha=alpha))
        AsicaCNNModel.add(MaxPooling2D((2, 2),padding='same'))
    elif numCapas>=3:
        AsicaCNNModel.add(Conv2D(numFilters*2, (kernel), activation='linear',padding='same'))
        AsicaCNNModel.add(BatchNormalization())
        AsicaCNNModel.add(LeakyReLU(alpha=alpha))
        AsicaCNNModel.add(MaxPooling2D((2, 2),padding='same'))
    elif numCapas>=4:
        AsicaCNNModel.add(Conv2D(numFilters*4, (kernel), activation='linear',padding='same'))
        AsicaCNNModel.add(BatchNormalization())
        AsicaCNNModel.add(LeakyReLU(alpha=alpha))                  
        AsicaCNNModel.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    AsicaCNNModel.add(Flatten())
    AsicaCNNModel.add(Dense(numFilters*4, activation='linear'))
    AsicaCNNModel.add(LeakyReLU(alpha=alpha))                  
    AsicaCNNModel.add(Dense(num_classes, activation=activacionCapaSalida))

    # Tasa de aprendizaje y optimizador
    learning_rate = 1e-5  # Cambia esto al valor que desees
  
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    AsicaCNNModel.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    AsicaCNNModel.summary()

    # stop training if there is no improvement in accuracy for 5 consecutive epochs
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4,restore_best_weights=True)
    mc = ModelCheckpoint(modelName, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)   
    # Entrenamiento. AQUÍ FALLA
    # print("Shape de train X: ", train_X.shape)
    # print("Shape de train_label: ", train_label.shape)
    # print("Shape de valid_X: ", valid_X.shape)
    # print("Shape de valid_label: ", valid_label.shape)

    Asica_train = AsicaCNNModel.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose="1",validation_data=(valid_X, valid_label),callbacks=[es,mc])
    
    # Save model
    fname=modelName+'.h5'
    # Guarda el modelo en formato h5
    AsicaCNNModel.save(fname)

    # print("Fashion history: ",fashion_train.history)
    accuracy = Asica_train.history['accuracy']
    # val_accuracy = Asica_train.history['val_accuracy']
    # loss = Asica_train.history['loss']
    # val_loss = Asica_train.history['val_loss']
    epochs = range(len(accuracy))

def iniciaCNN1(modelo,trainFolderData,kernel,numCapas,numEspectros,activacionCapaSalida,numFilters,indicesMFCC,kernel1x1, num_crossval=0):

    # Obtener la ruta del directorio actual
    baseFolder=os.getcwd()

    rutaModelo="cnnModels/cnn_"+modelo

    numMFCC=39  # 13
    numChannels=2
    numClasses=2

    rutaCompletaModelo=baseFolder+"/"+rutaModelo

    main(rutaCompletaModelo,trainFolderData,numMFCC,numEspectros,numChannels,numClasses,kernel1x1,kernel,numCapas,activacionCapaSalida,numFilters,indicesMFCC, num_crossval)
    return 0
