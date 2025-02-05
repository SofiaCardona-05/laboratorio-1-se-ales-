
# LABORATORIO 1 procesamiento digital de señales
# Análisis de Señales ECG y Ruido

## Introducción
Este programa permite analizar señales de electrocardiograma (ECG) y evaluar el impacto del ruido en ellas. Además, calcula la relación señal-ruido (SNR) para medir la calidad de la señal. Se incluyen diferentes tipos de ruido: gaussiano, de impulso y artefactos.

---

## ¿Qué es una Señal ECG?

El electrocardiograma (ECG) es una señal eléctrica generada por la actividad del corazón. Se mide mediante electrodos colocados en la piel y se representa como una serie de ondas que reflejan el ciclo cardíaco.

# *Información de la Señal Utilizada*

Origen: Registro MIT-BIH Arrhythmia Database (mitdb/100)

Duración: 10 segundos

Frecuencia de muestreo: 360 Hz

Intervalo de muestreo: 0.00278 segundos

Canal registrado: Derivación V5

Unidades: Milivoltios (mV)

Conversión de unidades: Se puede realizar con la función rdmat.m del toolbox WFDB-MATLAB.
---

## Tipos de Ruido en ECG

1. **Ruido Gaussiano**: Ruido aleatorio que se distribuye de manera normal y afecta la señal de forma continua.
2. **Ruido de Impulso**: Son picos repentinos que ocurren por interferencias en la señal.
3. **Ruido de Artefacto**: Alteraciones producidas por el movimiento del paciente o problemas en los electrodos.

---

## Relación Señal-Ruido (SNR)
El SNR (Signal-to-Noise Ratio) mide la cantidad de señal útil en comparación con el ruido presente. Se expresa en decibeles (dB) y se calcula como:

\[ SNR = 10 \log_{10} \left( \frac{P_{señal}}{P_{ruido}} \right) \]

Donde \( P_{señal} \) es la potencia de la señal original y \( P_{ruido} \) es la potencia del ruido añadido.

Valores de SNR:
- **Mayor a 20 dB**: Señal con poco ruido, buena calidad.
- **Entre 10 dB y 20 dB**: Señal con interferencias moderadas.
- **Menor a 10 dB**: Señal con alta contaminación de ruido, difícil de analizar.

---

## Funcionamiento del Código

### 1. Importación de Bibliotecas
Se importan las bibliotecas necesarias para la manipulación de datos y visualización:
```python
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
from scipy import stats
```

### 2. Carga de la Señal ECG
Se carga un archivo de datos en formato `.mat` que contiene la señal ECG:
```python
x = loadmat('100m.mat')
ecg = x['val']
ecg = np.transpose(ecg)
señal, datos = wfdb.rdsamp('100')
```

### 3. Extracción de Parámetros
Se obtienen los datos relevantes como la longitud de la señal, la frecuencia de muestreo y el canal de interés.
```python
n = datos['sig_len']
c = señal[:, 1]
fr = datos['fs']
```

### 4. Análisis Estadístico de la Señal
Se calcula la media, la desviación estándar y el coeficiente de variación:
```python
media = np.mean(señal)
d_e = np.std(señal)
cv = d_e / media
```

### 5. Visualización de la Señal Original
Se generan gráficos para analizar la forma de la señal ECG:
```python
plt.plot(ts, ecg)
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [mV]")
plt.show()
```

### 6. Generación de Ruido Gaussiano
Se agrega ruido gaussiano a la señal original y se calcula la SNR:
```python
ruido = np.random.normal(0, d_e, señal.shape)
señal_ruidosa = señal + ruido
Pseñal = np.mean(señal**2)
Pruido = np.mean(ruido**2)
SNR = 10 * np.log10(Pseñal / Pruido)
```

### 7. Generación de Ruido de Impulso
Se introduce ruido de impulso y se evalúa su impacto en la SNR:
```python
probabilidad_ruido = 0.1
mascara_ruido = np.random.rand(*señal.shape) < probabilidad_ruido
señal_ruidosai = señal.copy()
señal_ruidosai[mascara_ruido] = np.random.choice([np.min(señal), np.max(señal)], size=mascara_ruido.sum())
Pruido_impulso = np.mean((señal_ruidosai - señal)**2)
SNR_impulso = 10 * np.log10(Pseñal / Pruido_impulso)
```

### 8. Generación de Artefactos
Se simula un artefacto transitorio en la señal:
```python
duracion_artefacto = int(0.1 * fr)
amplitud_artefacto = 2 * np.max(señal)
artefacto = amplitud_artefacto * np.exp(-np.arange(duracion_artefacto) / (duracion_artefacto / 5))
posicion_artefacto = int(0.5 * len(señal))
señal_ruidosaa = señal.copy()
señal_ruidosaa[posicion_artefacto:posicion_artefacto + duracion_artefacto] += artefacto
Pruido_artefacto = np.mean((señal_ruidosaa - señal)**2)
SNR_artefacto = 10 * np.log10(Pseñal / Pruido_artefacto)
```

---

## Ejecución del Código
Para ejecutar el código en su entorno local:
1. Instale las bibliotecas necesarias:
   ```bash
   pip install numpy matplotlib wfdb scipy
   ```
2. Descargue el código y ejecútelo:
   ```bash
   python script.py
   ```

---
