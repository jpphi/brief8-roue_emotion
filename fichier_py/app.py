#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:13:33 2020

@author: jpphi
"""

import dash

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
