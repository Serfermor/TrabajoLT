import tweepy
import sys
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream
from tweepy.streaming import StreamListener
import pandas as pd
import csv

#provincia, coordenadas = next(entrada)  # Leer campos
# print(provincia, coordenadas)  # Mostrar campos
#Bibliotecas pagina web
from flask import Flask, render_template, flash
from flask import request
from flask import Flask, redirect, url_for, request
from wtforms import StringField, FieldList, FormField, SelectField


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
    csvLocalizacion=".//static/txt/provincias2.csv"
    with open(csvLocalizacion, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            x = row[0].split("|", 1)
            if localizacion == x[0]:
                return(x[1])

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

@app.route('/resultados') #creamos ruta para la pagina "contacto"
def resultados():
    return render_template('resultados.html') #Importamos archivo contact.html

@app.route('/busqueda', methods = ['POST', 'GET']) #creamos ruta para la pagina "resultados"
def busqueda():

        select = request.form.get('comp_select')
       
       #return str(select) # just to see what select is
        tema_elegido=""
        texto_prueba=""

        if request.method == 'POST':
            nombre = request.form['nombre']
            Localizacion = request.form['Localizacion']
            tema_elegido="Su búsqueda elegida es: " +nombre + " en " + Localizacion

            Localizacion = loadCoordenadasByCsv(Localizacion)
            texto_prueba = loadtweets(nombre,Localizacion)
           
        else:              
                txtfile=".//static/txt/info.txt"
                f = open (txtfile,'wb')
                f.close()
     
        return render_template('busqueda.html',texto_prueba=texto_prueba,tema_elegido=tema_elegido) #Importamos archivo resultados.html
 
if __name__ == '__main__': 
    app.run(debug=True) #nos permite ejecutar nuestra aplicacion

#!/usr/bin/env python





