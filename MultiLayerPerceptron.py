import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from keras import utils as utls
from tensorflow.keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten

# Definição de Hiperparâmetros
imageRows, imageCols, cores = 32, 32, 3
batchSize = 64
numClasses = 10
epochs = 5

# Carrega o dataset CIFAR-10
(XTreino, yTreino), (XTeste, yTeste) = cifar10.load_data()

# Normaliza os dados
XTreino = XTreino / 255.0
XTeste = XTeste / 255.0
yTreino = utls.to_categorical(yTreino, numClasses)
yTeste = utls.to_categorical(yTeste, numClasses)

inputShape = (imageRows, imageCols, cores)

# Modelo MLP
model = Sequential()

# Flatten da imagem para entrada da rede neural
model.add(Flatten(input_shape=inputShape))  # Flatten a imagem 32x32x3 para 1D

# Primeira camada densa
model.add(Dense(512, activation='relu'))  # Camada densa com 512 neurônios

# Segunda camada densa
model.add(Dense(512, activation='relu'))  # Segunda camada densa com 512 neurônios

# Camada de saída
model.add(Dense(numClasses, activation='softmax'))  # Camada de saída com 10 neurônios (para 10 classes)

# Resumo do modelo
model.summary()

# Compilação do modelo
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
minhaMLPModel = model.fit(XTreino, yTreino, batch_size=batchSize, epochs=epochs, validation_data=(XTeste, yTeste))

# Nomes das classes
nomeDosRotulos = ["avião", "carro", "pássaro", "gato", "cervo", "cachorro", "sapo", "cavalo", "navio", "caminhão"]

# Previsões e relatório de classificação
predicao = model.predict(XTeste)
print(classification_report(yTeste.argmax(axis=1), predicao.argmax(axis=1), target_names=nomeDosRotulos))

# Gráfico de acurácia
f, ax = plt.subplots()
ax.plot(minhaMLPModel.history['accuracy'], 'o-')
ax.plot(minhaMLPModel.history['val_accuracy'], 'x-')
ax.legend(['Acurácia no Treinamento', 'Acurácia na Validação'], loc=0)
ax.set_title('Treinamento e Validação - Acurácia por Época')
ax.set_xlabel('Época')
ax.set_ylabel('Acurácia')
