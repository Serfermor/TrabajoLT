# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:32:56 2020

Script para conseguir a través de a API de twitter y la librería tweepy, diferentes
tweets relacionados con el covid. De una localización y fecha dada.

Tal y como está planteada la API existen distintas versiones que permiten diferentes
funcionalidades.
Nosotros usaremos la versión más humilde (gratis), pues aunque podamos acceder durante 
cierto tiempo a las premium, existen ciertas restricciones que conplican todo el trabajo,
además de no poder filtrar por geolocalización.

Esta versión solo permite obtener tweets de una semana atrás respecto la fecha actual.


@author: serfe
"""


import tweepy
import json 
import Funciones
import numpy as np
from datetime import date


#######################################################################################

#Ya se ha comentado solo podemos realizar peticiones a la API, para tweets de hace una 
#semana

Fecha = date.today()    #Tomamos la fecha de hoy

Dia = 29            #Tener en cuenta al elegir los valores que solo 7 días atrás
Mes = 12
Año = 2020

Fecha_peticiones = Fecha # Obtenemos la fecha de la que obtendremos los tweets
Fecha_peticiones = Fecha_peticiones.replace(day = Dia + 1)
Fecha_peticiones = Fecha_peticiones.replace(month = Mes)
Fecha_peticiones = Fecha_peticiones.replace(year = Año)

Fecha_tweets = Fecha
Fecha_tweets = Fecha_tweets.replace(day=Dia)



#######################################################################################

#En primer lugar preparo las credenciales
CONSUMER_KEY="9OdHWr5K7HD6uF0zPDz0BRH40"
CONSUMER_SECRET="dkZTmit6HGfiJyhKPz7qs8rc9vm6KfvS3OnR8NRFLxuOIHUnCw"
ACCESS_TOKEN = "1328368975198171136-1Dt7GiGX5mnJs3EyuGPNqxzkiKHqcT"
ACCESS_TOKEN_SECRET = "Im1TQIgk5rR9e0jZSt9Mx6OEcddO8CmIOrAkD7r0xy3z8"

#Instancion el objeto OAuthHandler
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
#Llamo al método
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
#Instancio API, los parámetros wait es por si superamos los request máximos (15min) para que el
#programa quede a la espera y nos avise.
api = tweepy.API(auth, wait_on_rate_limit=True,
          wait_on_rate_limit_notify=True)

Lista=[]
Dates=[]

#Los ID en twitter actualmente ronda los 10e20, así para inicialización pongo un
#número mayor
ID=10e30 

# En aux guardamos el dia del que estamos tomando los datos, para que se ejecute 
# el bucle siempre que ambos coincidan, continuamente se iran tomando los dias de los tweets
aux=Dia
 

#Contador para prevenir que realicemos peticiones sin respuesta y estemos esperando
#indefinidamente
contador=0

# Utilizamos el cursor para poder conseguir más datos olvidándonos de la paginación que realiza twitter
# A este le pasamos el método al que queremos llamar y a continuación sus parámetros. 
# Con .items decimos cuantos items (tweets en este caso) queremos que nos proporcione.

while (aux==Dia)&(contador<10):  
    
    for tweet in tweepy.Cursor(api.search, q='covid',geocode="37.18817,-3.60667,100km",
                                       tweet_mode="extended",lang='es',
                                       until=Fecha_peticiones,max_id=ID).items(400):
        
        if (not tweet.retweeted) and ('RT @' not in tweet._json["full_text"]):
            Dates.append(tweet.created_at)
            ID=tweet.id-2
            tuit = tweet._json["full_text"]
            tuit = Funciones.Procesado_Tweet(tuit)
            Lista.append(tuit)
            
    aux = Dates[-1].day
    contador = contador+1    
    
    
#%% 
#Tal y como se plantea la extracción de datos si dentro del cursor los tweets cambian de
#día estos se leen en este bucle nos deshacemos de estos

while not (Dates[-1].day == Dia):
    Dates.pop()
    Lista.pop()



#%%
#Google proporciona una API como traductor que se llama googletrans, pero esta API
#en diciembre de 2020 está mostrando una serie de errores, respecto a la clase 
#Translator que es necesaria para su uso. Por ello la libreria que utilizo es una 
#modificación de esta por el usuario de github "lushan88a". La cual parece resolver
#este problema.

#La forma de uso es análoga, creamos un objeto de la clase translator, y con el 
#método translate, pasamos el string, el lenguaje original y al que vamos a traducir
#sino ponemos nigún idioma siempre traduce el texto al inglés

from google_trans_new import google_translator  

translator = google_translator() #<--- clase google_translator


List=[]

for tweet in Lista:
    result = translator.translate(tweet, lang_src='es', lang_tgt='en')
    List.append(result)



#%%
#Libreria TextBlob para sentimental analysis, se basa en el uso del algoritmo de
#naïve Bayes. La idea es introducir un string en inglés y este nos devuelve
#un parámetro de Polaridad (de -1 a +1) y Subjetividad (de 0 a +1). 

from textblob import TextBlob

Polaridad=[]
Subjetividad=[]

for tweet in List:
    a, b = TextBlob(tweet).sentiment
    Polaridad.append(a)
    Subjetividad.append(b)


#%% Hasta ahora hemos conseguido una listas de tweets en español, la cual hemos traducido
#a inglés, y de esta hemos obtenido dos nuevas listas con los parámetros de polaridad
#y subjetividad. Adicionalmente tenemos otra lista de datetimes, con la fecha de cada tweet

#Para el entrenamiendo de la red neuronal que se encargará de la predicción, debemos
#definir un intervalo de tiempo y dividir el día entre este. Dentro de cada intervalo
#realizamos la media para los parámetros que predicirá (Polaridad y subjetividad).

#Elegimos un intervalo de 2 horas, pues resulta cómodo y existen horas de las que no 
#se reciben tweets según se ha comprobado

aux=Dates[0].hour
suma1=0
suma2=0
z=0
k=0
Media_Pol=np.zeros(12)
Media_Sub=np.zeros(12)

for i in range(len(Dates)):
    
    if (Dates[i].hour>=aux-1):        
        suma1=suma1+Polaridad[i]
        suma2=suma2+Subjetividad[i]
        z=z+1.0
        
    else: 
        Media_Pol[k]=suma1/z
        Media_Sub[k]=suma2/z
        suma1=Polaridad[i]
        suma2=Subjetividad[i]
        z=1.0        
        k=k+1
        
        if (Dates[i].hour==aux-2) or (Dates[i].hour==aux-3):                       
            aux=aux-2
            
        else:
            aux2 =  aux - Dates[i].hour 
            aux2 = int(np.floor(aux2/2)-1)
            
            for j in range(aux2):
                Media_Pol[k+j]=0
                Media_Sub[k+j]=0
            
            aux=Dates[i].hour
            k = k+aux2


Media_Pol[k]=suma1/z
Media_Sub[k]=suma2/z     
        
        
        
        
#%% Creo los archivos Txt que contienen los tweets y las medidas realizadas sobre ellos

with open('{}.txt'.format(str(Fecha_tweets) + '_Tweets'),'w') as f:
    f.write('\nTweets sobre Covid Granada Fecha: ' + str(Fecha_tweets) + '\n\n\n')
    for i in range(len(List)):
        f.write("A las {}:{} horas:\n".format(Dates[i].hour,Dates[i].minute))
        f.write("%s\n\n" % json.dumps(Lista[i]))
        

with open('{}.txt'.format(str(Fecha_tweets) + '_Polarity'),'w') as f:
    f.write('\nMedia de la polaridad de los tweets referentes al Covid' 
            +' cada 2 horas en Granada Fecha: '
            + str(Fecha_tweets) + '\n\n\n')
    for item in Media_Pol:
        f.write("%s\n\n" % json.dumps(item))


with open('{}.txt'.format(str(Fecha_tweets) + '_Subjectivity'),'w') as f:
    f.write('\nMedia de la subjetividad de los tweets referentes al Covid' 
            +' cada 2 horas en Granada Fecha: '
            + str(Fecha_tweets) + '\n\n\n')
    for item in Media_Sub:
        f.write("%s\n\n" % json.dumps(item))
        