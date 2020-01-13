
import os
import sys
import pandas as pd
from os.path import join as JP

import dash
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import dash_core_components as dcc
import dash_html_components as html
from plotly.tools import mpl_to_plotly
from dash.dependencies import Input, Output

os.chdir('/app/')
print('WORKING DIRECTORY: ', os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(JP(os.getcwd(),'utils'))
sys.path.append(JP(os.getcwd(),'scripts'))

from utils.general import parse_yaml, ensure_directories
from scripts.algorithms.clustering import plot_centroids_as_wordclouds
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

# embedding_df['num_clusters'] = embedding_df['num_clusters'].apply(lambda x: str(x))

''' ---------------------------------- APPLICATION ---------------------------------- '''

app = dash.Dash()
server = app.server

app.layout = html.Div([

    # TOP BAR
    html.Div([
        html.H1('BASF')
    ], className = 'row', style={
        'backgroundColor':'green', 
        'text-align':'center', 
        'padding':'20px',
        'font-family':'verdana',
        'font-size':'150%',
        'color':'white'}),    

    # 3d Scatter Plot
    html.Div([
        html.H3('How many clusters would you like?'),
        dcc.Dropdown(
            id='num_cluster_dropwdown',
            options=[{'label': i, 'value': i} for i in embedding_df['num_clusters'].unique()],
            value=embedding_df['num_clusters'].unique()[0])
    ], style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "33%"}),

    html.Div([
        dcc.Graph(id='umap')
    ]),

    html.Div([
        dcc.Graph(id='wordclouds')
    ])
    
])

@app.callback(
    Output('umap','figure'), 
    [Input('num_cluster_dropwdown', 'value')])
def update_umap(num_clusters):
    
    d = embedding_df[embedding_df['num_clusters'] == num_clusters]    
    # trace = [go.Scatter3d(
    #     x = embedding_df['d1'], y = embedding_df['d2'], z = embedding_df['d3'],
    #     mode = 'markers', marker = {'size': 3 , 'color': embedding_df['cluster']}
    # )]

    # layout = go.Layout(
    #     title='Clustering of documents',
    #     hovermode = 'closest')

    # fig = go.Figure({'data':trace, 'layout': layout})

    fig = px.scatter_3d(d, x='d1', y='d2', z='d3', color='cluster')
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=700)
    return fig 

@app.callback(
    Output('wordclouds','figure'),
    [Input('num_cluster_dropwdown', 'value')])
def update_wordclouds(num_clusters):
    d = words_df[words_df['num_clusters'] == num_clusters]    
    print(words_df.columns)
    print(words_df.shape)
    print(num_clusters)
    print(words_df[words_df['num_clusters' == num_clusters]].shape)
    return mpl_to_plotly(
        plot_centroids_as_wordclouds(
            words_df[words_df['num_clusters' == num_clusters]], 
            n_cols=3))



if __name__ == '__main__':
    app.run_server(debug=True)

exit()