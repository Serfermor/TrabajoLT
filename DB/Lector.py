# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 16:23:26 2020


SCRIPT para tomar los datos de los .txt

@author: serfe
"""

from datetime import date
import Funciones

#############################################################################

#Elegir la fecha del archivo que queremos importar

Fecha = date.today()

Dia = 23
Mes =12
Año = 2020



Fecha_tweets = Fecha
Fecha_tweets = Fecha_tweets.replace(day=Dia)
Fecha_tweets = Fecha_tweets.replace(month=Mes)
Fecha_tweets = Fecha_tweets.replace(year=Año)


############################################################################


Aux1=[]
Aux2=[]
Media_Pol=[]
Media_Sub=[]

Aux1 = [line.rstrip('\n') for line in open('{}.txt'.format(str(Fecha_tweets)
                                                               + '_Polarity'))]


Aux2 = [line.rstrip('\n') for line in open('{}.txt'.format(str(Fecha_tweets)
                                                               + '_Subjectivity'))]


for i in range(len(Aux1)):
    if Funciones.isFloat(Aux1[i]):
        Media_Pol.append(Aux1[i])



for i in range(len(Aux2)):
    if Funciones.isFloat(Aux2[i]):
        Media_Sub.append(Aux2[i])
    

# Las variables "Pol" y "Subj" contienen las medias de las correspondientes medidas
# cada intervalo de 2horas.


#%%

a = Polari