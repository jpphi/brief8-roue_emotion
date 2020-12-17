#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:45:30 2020

@author: jpphi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:41:54 2020

@author: jpphi
Analyse en composante principal
"""

#---------------------------------------------------------------------------------------------------------------------------------
#                                               Liste des imports et constantes
#---------------------------------------------------------------------------------------------------------------------------------

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash

from app import app

import pandas as pd
import plotly.express as px

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV


# Constantes pour lisibilité du code
EMOTION_FINAL= 1
TEXT_EMOTION= 2
PAS_DE_FICHIER= 0

#---------------------------------------------------------------------------------------------------------------------------------
#               Chargement des fichiers, créations des datas frames et dictionnaires
#---------------------------------------------------------------------------------------------------------------------------------

# On charge les 2 fichiers. Ce qui pose le problème de la place mémoire, mais permet d'aller plus vite dans les affichages 
df_ef = pd.read_csv("./data/Emotion_final.csv")
df_te = pd.read_csv("./data/text_emotion.csv")

# tokénisation avec création de nouvelles colonnes dans les datas frames
corpus=[]
for el in df_ef.Text:
    token= el.split()
    corpus.append(token)
df_ef['token_brut'] = pd.Series(corpus)
#print(pd.Series(corpus).head(10))

corpus=[]
for el in df_te.content:
    token= el.split()
    corpus.append(token)
df_te['token_brut'] = pd.Series(corpus)
#print(pd.Series(corpus).head(10))

# Création des dictionnaire de mots
dict_ef={}
for liste in df_ef.token_brut:
    for mot in liste:
        if dict_ef.get(mot):
            dict_ef[mot]+= 1
        else:
            dict_ef[mot]= 1

liste_mot= sorted(dict_ef.items(), key=lambda t: t[1],reverse=True)
mots_ef=[]
occur_ef=[]
for el in liste_mot:
    mots_ef.append(el[0])
    occur_ef.append(el[1])

dict_te={}
for liste in df_ef.token_brut:
    for mot in liste:
        if dict_te.get(mot):
            dict_te[mot]+= 1
        else:
            dict_te[mot]= 1

liste_mot= sorted(dict_ef.items(), key=lambda t: t[1],reverse=True)
mots_te=[]
occur_te=[]
for el in liste_mot:
    mots_te.append(el[0])
    occur_te.append(el[1])

fichier= PAS_DE_FICHIER

#---------------------------------------------------------------------------------------------------------------------------------
#                                                   Layout de la page
#---------------------------------------------------------------------------------------------------------------------------------

layout = html.Div([
    html.H3('Analyse des données sous forme graphique.'),
    html.H4("Note: Améliorer fct update_figure pour réactualiser affichage sur chargement de l'autre fichier."),
    html.H4("Note: Attention au temps de chargement des fichiers..."),

    html.Button('Emotion finale !?!', id='chargementp2_1', n_clicks=0),
    html.Button('Texte émotion %-)', id='chargementp2_2', n_clicks=0),

    dcc.Dropdown(
        id='app-2-dropdown',
        options=[
            {'label': 'Page 2 - {}'.format(i), 'value': i} for i in [
                'Selectionner un graph', 'Occurence des emotions', 'Occurence des mots', 'Graph 3'
            ]
        ],
        value= 'Selectionner un graph'
    ),

    html.Div(id='container_traitement'),
    html.Div(id='container_button_timestamp_p2'),
    
    dcc.Graph(id='graph2'),

    dcc.Link('Retour à la page 1.', href='/pages/page1')

    #html.Div("Chargement demandée",id='container-button-visu'),
    #html.Div("Chargement demandée",id='container-button-basic'),
 
])

#---------------------------------------------------------------------------------------------------------------------------------
#                                                           Callbacks
#---------------------------------------------------------------------------------------------------------------------------------

@app.callback(Output('container_button_timestamp_p2', 'children'),
              Input('chargementp2_1', 'n_clicks'),
              Input('chargementp2_2', 'n_clicks'))
def displayClick(btn1, btn2):
    global fichier, df_ef, df_te

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'chargementp2_1' in changed_id:




        fichier= EMOTION_FINAL

        dftexte= df_ef.Text
        dfemotion= df_ef.Emotion

    elif 'chargementp2_2' in changed_id:
        #sentiment author content
        dftexte= df_te.content
        dfemotion= df_te.sentiment
        fichier= TEXT_EMOTION
        msg= "Fichier text_emotion.csv chargé"

    else:
        fichier= PAS_DE_FICHIER
        msg = "Aucun fichier n'a encore été chargé."

    if fichier!= PAS_DE_FICHIER:
        vectorisation_cv = CountVectorizer()
        cv= vectorisation_cv.fit_transform(dftexte)
        X_train, X_test, y_train, y_test = train_test_split(cv, dfemotion, test_size = 0.3,random_state=42)

        logreg= LogisticRegression(max_iter=100)
        logreg.fit(X_train,y_train)
        pred_logreg=logreg.predict(X_test)

        res= classification_report(y_test,pred_logreg, output_dict=True)

        dframe = pd.DataFrame(res).transpose()
        dframe["Libellé"] = dframe.index
        msg= html.Table([
        html.Thead(
            #html.Tr([html.Th("Emotion"),html.Th("precision"),html.Th("recall"),html.Th("f1-score"),html.Th("support")])
            html.Tr([html.Th(col) for col in dframe.columns])
        ),
        html.Tbody([
            html.Tr([html.Td(dframe.iloc[i][col]) for col in dframe.columns
            ]) for i in range(0, len(dframe))
            ])
        ]) 






        
    return msg

@app.callback(
    Output('graph2', 'figure'), 
    [Input('app-2-dropdown', 'value'), Input('chargementp2_1', 'n_clicks'), Input('chargementp2_2', 'n_clicks')]
    )

#    Input('year-slider', 'value'))
def update_figure(value, click1, click2):
    global fichier
    """
    print("click: ", click1, click2)
    if click1== 1:
        click1= 0
        fichier= EMOTION_FINAL
    elif click2== 1:
        click1= 2
        fichier= TEXT_EMOTION
    else:
        fichier= PAS_DE_FICHIER
    """
    print("clic et fich", click1, click2, fichier)


    if fichier== PAS_DE_FICHIER: # rien à afficher!
        fig2 = px.scatter()
        return fig2

    elif fichier== EMOTION_FINAL:
        # pour l'histogramme du Graph 1
        df= df_ef
        abscisse= df_ef.Emotion
        couleur= df_ef.Emotion
        ancre= df_ef.Emotion

        # Pour le graph d'occurence des mots
        x= mots_ef
        y= occur_ef

        #ordonnee= df_ef.Emotion

    else: # TEXT_EMOTION
        df=df_te
        abscisse= df_te.sentiment
        couleur= df_te.sentiment
        ancre= df_te.sentiment

        # Pour le graph d'occurence des mots
        x= mots_te
        y= occur_te

        #ordonnee= df.author

    if value== 'Selectionner un graph': # QUE RETOURNE POUR NE RIEN AFFICHER ?
        fig2 = px.scatter()
        return fig2

    elif value== "Occurence des emotions":
        fig2 = px.histogram(df, x= abscisse, color= couleur, hover_name= ancre).update_xaxes(categoryorder="total descending")
        return fig2

    elif value== "Occurence des mots": #df_te['token_brut']
        fig2 = px.scatter(df, x= x, y= y, hover_name= y, log_y= True)
        return fig2

    else :
        fig2 = px.scatter()
        return fig2
