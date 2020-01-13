
import os
import sys
import pandas as pd
from os.path import join as JP

import dash
import plotly.graph_objs as go
import plotly.figure_factory as ff
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

os.chdir('/app/')
print('WORKING DIRECTORY: ', os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(JP(os.getcwd(),'utils'))
sys.path.append(JP(os.getcwd(),'scripts'))

from utils.general import parse_yaml, ensure_directories
config = parse_yaml('config.yaml')
paths = config['paths']


''' ---------------------------------- HP ---------------------------------- '''

MIN_K = 2
MAX_K = 10

''' ---------------------------------- DATA ---------------------------------- '''

words_df = pd.read_csv(
    JP(paths['results'], 'words_per_cluster_{}_to_{}.csv'.format(MIN_K,MAX_K)), index_col=0)

embedding_df = pd.read_csv(
    JP(paths['results'], 'embeddings_per_cluster_{}_to_{}.csv'.format(MIN_K,MAX_K)), index_col=0)

''' ---------------------------------- APPLICATION ---------------------------------- '''

app = dash.Dash()
server = app.server

app.layout = html.Div([

    # TOP BAR
    html.Div([
        html.H1('BASF')
    ])    
    
], className = 'row', style={'backgroundColor':'green', 'text-align':'center', 'padding':'20px'})

if __name__ == '__main__':
    app.run_server(debug=True)

exit()