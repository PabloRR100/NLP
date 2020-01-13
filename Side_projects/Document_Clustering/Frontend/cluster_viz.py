
import os
import sys
import glob
import flask
import pickle
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
image_directory = paths['images']
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
print('Found {} images'.format(len(list_of_images)))

''' ---------------------------------- HP ---------------------------------- '''

MIN_K = 2
MAX_K = 10


''' ---------------------------------- DATA ---------------------------------- '''

images_basename = 'wordcloud_clusters'

words_df = pd.read_csv(
    JP(paths['results'], 'words_per_cluster_{}_to_{}.csv'.format(MIN_K,MAX_K)), index_col=0)

embeddings_df = pd.read_csv(
    JP(paths['results'], 'embeddings_per_cluster_{}_to_{}.csv'.format(MIN_K,MAX_K)), index_col=0)

with open(JP(paths['results'], 'words_per_cluster_{}_to_{}.pkl'.format(MIN_K,MAX_K)), 'rb') as f:
    words = pickle.load(f)

with open(JP(paths['results'], 'embeddings_per_cluster_{}_to_{}.pkl'.format(MIN_K,MAX_K)), 'rb') as f:
    embeddings = pickle.load(f)

# embeddings_df['num_clusters'] = embeddings_df['num_clusters'].apply(lambda x: str(x))

''' ---------------------------------- APPLICATION ---------------------------------- '''

app = dash.Dash()
server = app.server
static_image_route = '/static/'

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
            options=[{'label': i, 'value': i} for i in embeddings_df['num_clusters'].unique()],
            value=embeddings_df['num_clusters'].unique()[0])
    ], style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "33%"}),

    html.Div([
        dcc.Graph(id='umap')
    ]),

    # # Wordcloud
    # html.Img(id='wordclouds')
    
])

@app.callback(
    Output('umap','figure'), 
    [Input('num_cluster_dropwdown', 'value')])
def update_umap(num_clusters):
    
    d = embeddings_df[embeddings_df['num_clusters'] == num_clusters]    
    # trace = [go.Scatter3d(
    #     x = embeddings_df['d1'], y = embeddings_df['d2'], z = embeddings_df['d3'],
    #     mode = 'markers', marker = {'size': 3 , 'color': embeddings_df['cluster']}
    # )]

    # layout = go.Layout(
    #     title='Clustering of documents',
    #     hovermode = 'closest')

    # fig = go.Figure({'data':trace, 'layout': layout})

    fig = px.scatter_3d(d, x='d1', y='d2', z='d3', color='cluster')
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=700)
    return fig 

# @app.callback(
#     Output('wordclouds','src'),
#     [Input('num_cluster_dropwdown', 'value')])
# def update_image_src(value):
#     print('New src = ', )
#     return static_image_route + value

# @app.server.route('{}<image_path>.png'.format(static_image_route))
# def serve_image(image_path):
#     image_name = '{}.png'.format(image_path)
#     print(image_name)
#     if image_name not in list_of_images:
#         raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
#     return flask.send_from_directory(image_directory, image_name)

# def update_wordclouds(num_clusters):
#     plotly_fig = mpl_to_plotly(
#         plot_centroids_as_wordclouds(
#             words[num_clusters], n_cols=3, show=False))
#     print('Hi there!')
#     return plotly_fig

if __name__ == '__main__':
    app.run_server(debug=True)

exit()