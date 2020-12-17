#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:41:54 2020

@author: jpphi
"""

#---------------------------------------------------------------------------------------------------------------------------------
#                                               Liste des imports et constantes
#---------------------------------------------------------------------------------------------------------------------------------

import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

import plotly.express as px

AFF_INTEGRAL_P1= False

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#text_emotion = pd.read_csv("./data/text_emotion.csv")
#Emotion_final = pd.read_csv("./data/Emotion_final.csv")

df= None
#---------------------------------------------------------------------------------------------------------------------------------
#                                                   Layout de la page
#---------------------------------------------------------------------------------------------------------------------------------

layout = html.Div([
    html.H3('Visualisation du contenu des fichiers csv.'),
    html.H4("Pour un affichage de l'intégralité des données, mettre à True la constante AFF_INTEGRAL dans page 1."),
    html.H4("Remplacer AFF_INTEGRAL_P1 par case à cocher."),
    html.Button('Emotion finale !?!', id='chargement_1', n_clicks=0),

    html.Button('Texte émotion %-)', id='chargement_2', n_clicks=0),

    html.Div(id='app-1-display-value'),

    #html.Div("Chargement demandée",id='container-button-visu'),
    #html.Div("Chargement demandée",id='container-button-basic'),
    html.Div("Zone chargement",id='container-button-timestamp'),

    #dcc.Graph(id='graph'),

    dcc.Link('Aller vers la page 2', href='/pages/page2')
    #,    classement(df2016)
])

#---------------------------------------------------------------------------------------------------------------------------------
#                                                           Callbacks
#---------------------------------------------------------------------------------------------------------------------------------

@app.callback(Output('container-button-timestamp', 'children'),
              Input('chargement_1', 'n_clicks'),
              Input('chargement_2', 'n_clicks'))
def displayClick(btn1, btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'chargement_1' in changed_id:
        # récupérer le fichier dans le répertoire data
        df = pd.read_csv("./data/Emotion_final.csv")
        if AFF_INTEGRAL_P1== True: max_rows= len(df)
        else : max_rows= len(df) // 20
        #n_clicks= 0
        msg= html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in df.columns])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(df.iloc[i][col]) for col in df.columns
                ]) for i in range(0, max_rows)
                ])
            ])

    elif 'chargement_2' in changed_id:
        # récupérer le fichier dans le répertoire data
        df = pd.read_csv("./data/text_emotion.csv")
        if AFF_INTEGRAL_P1== True: max_rows= len(df)
        else : max_rows= len(df) // 20
        msg= html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in df.columns])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(df.iloc[i][col]) for col in df.columns
                    ]) for i in range(0, max_rows)
                    ])
                ])
    
    else:
        msg = "Aucun fichier n'a été chargé."
    
    return html.Div(msg)











"""
@app.callback(
    Output('graph', 'figure'),
    Input('app-1-dropdown', 'value'))
#    Input('year-slider', 'value'))
def update_figure(value):
    if value== "research":
    #filtered_df = df[df.year == selected_year]
        fig = px.scatter(df2016, x="world_rank", y="research", color="country", hover_name="country")
        return fig
    elif value== "income":
        fig = px.scatter(df2016, x="world_rank", y="income", color="country", hover_name="country")
        return fig

    else :
        fig = px.scatter(df2016, x="world_rank", y="teaching", color="country", hover_name="country")
        return fig
"""

"""
@app.callback(
    Output('app-1-display-value', 'children'),
    Input('app-1-dropdown', 'value'))
def display_value(value):
    if value== "Graph 1":
        return "Graph1 à afficher"
    if value== "Graph 2":
        return "Graph 2 à afficher"
    if value== "Graph 3":
        return "Graph 3 à afficher"
    return 'Pas de graphe sélectionné.'
#    return 'Valeur sélectionnée "{}"'.format(value)
"""
