
import os
import sys
import pandas as pd
from os.path import join as JP
from utils.general import parse_yaml, ensure_directories

import dash
import plotly.graph_objs as go
import plotly.figure_factory as ff
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

os.chdir('/app/')
print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(JP(os.getcwd(),'utils'))
sys.path.append(JP(os.getcwd(),'scripts'))

results_path = os.path.abspath('../results/dicts/ResNet110')
