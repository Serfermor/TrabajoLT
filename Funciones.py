# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:45:22 2020

@author: serfe
"""


import re


def Procesado_Tweet(tweet):
    #Quito caracteres de HTML
    tweet = re.sub(r'\&\w*;','',tweet)
    #Quito Hyperlinks
    tweet = re.sub(r'https?:\/\/.*\/\w*','',tweet)
    #Paso a minúscuka
    tweet = tweet.lower()
    #Quito los usuarios
    tweet = re.sub(r'@[^\s]+','',tweet)
    #Quito caracteres fuera de BMP de Unicode
    tweet = ''.join(e for e in tweet if (e.isalnum() or e.isspace()))
    #Quito varios espacios consecutivos
    tweet = re.sub("\s\s+" , " ", tweet)
    #Quito espacios iniciales
    tweet = tweet.lstrip(' ')    
    return tweet




def isFloat(string):
    try:
        float(string)
        return True
    except:
        return False