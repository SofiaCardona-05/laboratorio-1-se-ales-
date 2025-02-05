
# -- coding: utf-8 --
"""
Created on Tue Jan 28 20:35:24 2025

@author: User
"""

# len() = es una funcion que devuelve la longitud de un objeto
# mean() = calcula la media de la señal
# std() = calcula la desviacion estandar
# percentile()


import numpy as np #Operaciones matematicas 
import matplotlib.pyplot as plt #Graficas y dibujos
import wfdb  #Para carga de datos y manipularlos 
from scipy.io import loadmat
from scipy import stats



x=loadmat ('100m.mat')
ecg=x['val']
ecg=np.transpose(ecg)

señal, datos = wfdb.rdsamp('100')
record = wfdb.rdrecord('100')



n = datos['sig_len']#llamar longitud de verctor datos
c = señal[:,1] #canal que vamos a usar 2
fr = datos ['fs'] #Frecuencia
#t = np.linspace(0,d, len(señal)) #crea un vector tiempo, empieza en 0, termina en d y se separa en puntos uniformes

kde = stats.gaussian_kde(c)

d = np.linspace(min(c),max(c),100)
p = kde(d)



tm=1/fr #tiempo de muestreo
t= np.linspace(0,np.size(señal),np.size(señal))*tm
ts= np.linspace(0,np.size(ecg),np.size(ecg))*tm
s=señal.flatten()

media = np.mean(señal)
mediap = sum(s)/np.size(s)
print("Media Programada: ",mediap)
print("Media Predefinida: ",media)
d_e = np.std(señal) #desviacion estandar
d_ep= np.sqrt(sum((x - mediap) ** 2 for x in s) / len(s))
print("Desviación estándar Programada:", d_ep)
print("Desviación Predefinida:",d_e)
cv = d_e/media
print("Coeficiente de variación:", cv)



plt.hist(señal, bins=100)
plt.xlabel("tiempo [s]")
plt.ylabel("voltaje [mV]")
plt.title("Histograma")
plt.show()

plt.plot(ts,ecg)
plt.xlabel("tiempo [s]")
plt.ylabel("voltaje [mV]")
plt.show()

plt.plot(d,p)
plt.xlabel("tiempo [s]")
plt.ylabel("voltaje [mV]")
plt.title("Funcion probabilidad")
plt.show()

#senal ruido gaussiano
ruido = np.random.normal(0, d_e, señal.shape)
señal_ruidosa = señal + ruido
# Convertir a una dimensión
señal_ruidosa = señal_ruidosa.flatten()
t = np.arange(señal_ruidosa.shape[0]) * tm 

plt.plot(t, señal_ruidosa) 

plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [mV]")
plt.title("Señal con Ruido Gaussiano")
plt.show()

#calcular SNR GAUSSIANO 
Pseñal = np.mean(señal**2)
Pruido = np.mean(ruido**2)
SNR = 10 * np.log10(Pseñal / Pruido)

print("SNR gaussiano:", SNR, "dB")

#senal ruido impulso
probabilidad_ruido = 0.1  # Ajusta este valor según la cantidad de ruido que desees
mascara_ruido = np.random.rand(*señal.shape) < probabilidad_ruido
señal_ruidosai = señal.copy()  # Crea una copia para no modificar la señal original
señal_ruidosai[mascara_ruido] = np.random.choice([np.min(señal), np.max(señal)], size=mascara_ruido.sum())
señal_ruidosai = señal_ruidosai.flatten()
t = np.arange(señal_ruidosai.shape[0]) * tm 
plt.plot(t, señal_ruidosai)
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [mV]")
plt.title("Señal con Ruido de Impulso")
plt.show()

# --- CALCULAR SNR IMPULSO ---
# Aplanar señal y mascara_ruido
señal_plana = señal.flatten()
mascara_ruido_plana = mascara_ruido.flatten()

  # 1. Estima la señal "limpia" restando el ruido de la señal ruidosa
señal_estimada_limpia = señal_ruidosai - (señal_ruidosai - señal_plana) * mascara_ruido_plana 
  # 2. Calcula la potencia de la señal original
Pseñal = np.mean(señal**2)
  # 3. Calcula la potencia del ruido de impulso
Pruido_impulso = np.mean((señal_ruidosai - señal_estimada_limpia)**2)
  # 4. Calcula la SNR
SNR_impulso = 10 * np.log10(Pseñal / Pruido_impulso)

print("SNR del ruido de impulso:", SNR_impulso, "dB")

#senal ruido artefacto 
duracion_artefacto = int(0.1 * fr)  # Duración en muestras (0.1 segundos en este ejemplo)
amplitud_artefacto = 2 * np.max(señal)  # Amplitud del artefacto
artefacto = amplitud_artefacto * np.exp(-np.arange(duracion_artefacto) / (duracion_artefacto / 5))  # Decaimiento exponencial, ruido pico repentino
posicion_artefacto = int(0.5 * len(señal))  # Posición del artefacto en la señal
señal_ruidosaa = señal.copy()
# Expandir las dimensiones de 'artefacto' para que coincidan con la sección de 'señal_ruidosaa'
artefacto_expandido = np.broadcast_to(artefacto[:, np.newaxis], (duracion_artefacto, señal.shape[1]))  
señal_ruidosaa[posicion_artefacto:posicion_artefacto + duracion_artefacto] += artefacto_expandido  
señal_ruidosaa = señal_ruidosaa.flatten()
t = np.arange(señal_ruidosaa.shape[0]) * tm 

plt.plot(t, señal_ruidosaa)
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [mV]")
plt.title("Señal con Artefacto")
plt.show()
# --- CALCULAR SNR ARTEFACTO ---
# 1. Estima la señal "limpia" restando el artefacto
señal_estimada_limpia = señal_ruidosaa.copy()
señal_estimada_limpia[posicion_artefacto:posicion_artefacto + duracion_artefacto] -= artefacto 
# 2. Calcula la potencia de la señal original
Pseñal = np.mean(señal**2)
# 3. Calcula la potencia del ruido de artefacto
Pruido_artefacto = np.mean((señal_ruidosaa - señal_estimada_limpia)**2)
# 4. Calcula la SNR
SNR_artefacto = 10 * np.log10(Pseñal / Pruido_artefacto)

print("SNR del ruido de artefacto:", SNR_artefacto, "dB")