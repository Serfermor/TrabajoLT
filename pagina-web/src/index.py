import tweepy
import sys
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream
from tweepy.streaming import StreamListener
import pandas as pd
import csv
import traceback


#Bibliotecas pagina web
from flask import Flask
from flask import render_template
from flask import request




CONSUMER_KEY="9OdHWr5K7HD6uF0zPDz0BRH40"
CONSUMER_SECRET="dkZTmit6HGfiJyhKPz7qs8rc9vm6KfvS3OnR8NRFLxuOIHUnCw"
ACCESS_TOKEN = "1328368975198171136-1Dt7GiGX5mnJs3EyuGPNqxzkiKHqcT"
ACCESS_TOKEN_SECRET = "Im1TQIgk5rR9e0jZSt9Mx6OEcddO8CmIOrAkD7r0xy3z8"

auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = API(auth, wait_on_rate_limit=True,
          wait_on_rate_limit_notify=True)

#Buscar tweets por una palabra clave

def load_infoCSV(csvfile, nombre,Localizacion):
    info = []
    for tweet in tweepy.Cursor(api.search, q=nombre,geocode=Localizacion,tweet_mode='extended').items(10):
        if 'retweeted_status' in tweet._json:
            info+=(tweet._json['retweeted_status']['full_text']+"\n\n")
        else:
            info+=tweet.full_text+"\n\n"
    info_df =(pd.DataFrame(info)).to_csv(csvfile, index = False, encoding = 'utf-8')

def loadtweets(nombre, Localizacion):
    info = ""
    for tweet in tweepy.Cursor(api.search, q=nombre,geocode=Localizacion).items(10):
        
        info+=  tweet.text + '\n\n'
    
    return info

def loadCoordenadasByCsv(localizacion):
    csvLocalizacion="./static/txt/provincias2.csv"
    with open(csvLocalizacion, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            x = row[0].split("|", 1)
            if localizacion == x[0]:
                return(x[1])
    return ''

#-----Pagina web-------
app = Flask(__name__) #Para confirmar que es el archivo que va a arrancar la aplicacion

@app.route('/') #creamos una ruta para la pagina principal
def home(): 

    return render_template('home.html') #para importar archivo html#

@app.route('/about') #creamos ruta para la pagina "acerca de"
def about():
    return render_template('about.html') #Importamos archivo about.html

@app.route('/contact') #creamos ruta para la pagina "contacto"
def contact():
    return render_template('contact.html') #Importamos archivo contact.html

@app.route('/resultados', methods = ['POST', 'GET']) #creamos ruta para la pagina "contacto"
def resultados():
    return render_template('resultados.html') #Importamos archivo contact.html

@app.route('/busqueda', methods = ['POST', 'GET']) #creamos ruta para la pagina "resultados"
def busqueda():

        tema_elegido=""
        sel_tweets=""
        if request.method == 'POST':
            nombre = request.form['nombre']
            Localizacion = request.form['Localizacion']
            tema_elegido="Su búsqueda elegida es: " +nombre + " en " + Localizacion

            Localizacion = loadCoordenadasByCsv(Localizacion)
            sel_tweets = loadtweets(nombre,Localizacion)

            if Localizacion == '':
                sel_tweets = 'La localización no es correcta. Revisela por favor.'
           
        else:              
            tema_elegido=""
            sel_tweets=""
    
     
        return render_template('busqueda.html',sel_tweets=sel_tweets,tema_elegido=tema_elegido) 
   

@app.route('/tweets') #creamos ruta para la pagina "contacto"
def tweets():
    id = request.args.get('fecha')
    return render_template('tweets.html', fecha_fichero=id) #Importamos archivo contact.html       
 
if __name__ == '__main__': 
    app.run(debug=True) #nos permite ejecutar nuestra aplicacion

#!/usr/bin/env python





