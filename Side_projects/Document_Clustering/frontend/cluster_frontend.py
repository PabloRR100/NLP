
import os
import sys
import base64
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

DOCKER = os.environ.get('IM_IN_DOCKER', False)
if DOCKER: 
    WORKDIR = '/app/'
else:
    WORKDIR = 'c:\\Users\\RUIZP4\\Documents\\DOCS\\Pablo_Personal\\StanfordNLP\\Side_projects\\Document_Clustering'
    WORKDIR = '/Users/pabloruizruiz/OneDrive/Courses/NLP_Stanford/Side_projects/Document_Clustering/'

os.chdir(WORKDIR)
print('WORKING DIRECTORY: ', WORKDIR)
sys.path.append(WORKDIR)
sys.path.append(JP(WORKDIR,'utils'))
sys.path.append(JP(WORKDIR,'scripts'))

from utils.general import parse_yaml, ensure_directories
from scripts.algorithms.clustering import plot_centroids_as_wordclouds
config = parse_yaml('config.yaml')
paths = config['paths']
image_directory = paths['images']
logo = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/BASF-Logo_bw.svg/1280px-BASF-Logo_bw.svg.png'



''' ---------------------------------- DATA ---------------------------------- '''

images_basename = 'wordcloud_clusters'
# words_df = pd.read_csv(
#     JP(paths['results'], 'words_per_cluster_{}_to_{}.csv'.format(MIN_K,MAX_K)), index_col=0)
# embeddings_df = pd.read_csv(
#     JP(paths['results'], 'embeddings_per_cluster_{}_to_{}.csv'.format(MIN_K,MAX_K)), index_col=0)
# with open(JP(paths['results'], 'words_per_cluster_{}_to_{}.pkl'.format(MIN_K,MAX_K)), 'rb') as f:
#     words = pickle.load(f)
# with open(JP(paths['results'], 'embeddings_per_cluster_{}_to_{}.pkl'.format(MIN_K,MAX_K)), 'rb') as f:
#     embeddings = pickle.load(f)
with open(JP(paths['results'], 'clustering.pkl'), 'rb') as f:
    results = pickle.load(f)

''' ---------------------------------- HP ---------------------------------- '''

VIZ_DIMENSIONS = [2,3]
K_VALUES = list(results.keys())
CLUST_METHODS = list(results[K_VALUES[0]].keys())
DIM_REDUCTORS = list(results[K_VALUES[0]][CLUST_METHODS[0]].keys())
# embeddings_df['num_clusters'] = embeddings_df['num_clusters'].apply(lambda x: str(x))

''' ---------------------------------- APPLICATION ---------------------------------- '''

app = dash.Dash(__name__,
    external_stylesheets=[
        'https://codepen.io/chriddyp/pen/bWLwgP.css'
    ]
)

server = app.server
static_image_route = '/static/'

app.layout = html.Div([

    # TOP BAR
    html.Div([
        html.A([
                html.Img(src=logo,
                     style={'max-width':'10%', 'max-height':'10%'})
            ], href='https://www.basf.com/es/es.html')
        # html.H1('BASF')
    ], className = 'row', style={
        'backgroundColor':'green', 
        'text-align':'center', 
        'padding':'20px',
        'font-family':'verdana',
        'font-size':'150%',
        'color':'white'}),    
    
    html.Div([
        
        html.Div([
            
            # 3d Scatter Plot
            html.Div([
                
                # html.Div([
                #     html.H3('How many clusters would you like?'),
                # ], style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%", 'text-align':'center'}),

                html.Div([
                
                    # Number of clusters ?
                    html.Div([
                        dcc.Dropdown(
                        id='num_cluster_dropwdown',
                        options=[{'label': i, 'value': i} for i in K_VALUES],
                        value=K_VALUES[0])
                    ], className='six columns'), # style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "30%"})

                    # Clustering method ? 
                    html.Div([
                        dcc.Dropdown(
                        id='cluster_alg_dropwdown',
                        options=[{'label': i, 'value': i} for i in CLUST_METHODS],
                        value=CLUST_METHODS[0])
                    ], className='six columns'), # style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "30%"})

                    # Dimensions to visualize ?
                    html.Div([
                        dcc.Dropdown(
                        id='viz_dim_dropwdown',
                        options=[{'label': i, 'value': i} for i in VIZ_DIMENSIONS],
                        value=3)
                    ], className='six columns'), # style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "30%"})

                    # Dimensionality reduction ?
                    html.Div([
                        dcc.Dropdown(
                        id='dim_reduction_dropwdown',
                        options=[{'label': i, 'value': i} for i in DIM_REDUCTORS],
                        value=DIM_REDUCTORS[0])
                    ], className='six columns') # style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "30%"})

                ], className='row')

            ]),

            html.Div([
                dcc.Graph(id='umap')
            ]),

        ], className = 'six columns'),

        # Wordcloud
        html.Div([
            html.Img(id='wordclouds', src='')
        ], className = 'six columns', style={}),

    ], className = 'row')
])

@app.callback(
    Output('umap','figure'), 
    [Input('num_cluster_dropwdown', 'value'),
     Input('cluster_alg_dropwdown', 'value'),
     Input('viz_dim_dropwdown', 'value'),
     Input('dim_reduction_dropwdown', 'value')])
def update_umap(num_clusters, clust_alg, viz_dim, dim_red):
    d = results[num_clusters][clust_alg][dim_red]['scatter']   
    d['cluster'] = d['cluster'].astype(str)
    if viz_dim == 3:
        fig = px.scatter_3d(d, x='d1', y='d2', z='d3', color='cluster')
    else:
        fig = px.scatter(d, x='d1', y='d2', color='cluster')
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(title_text='Document embeddings', title_x=0.5, showlegend=False)
    return fig 

@app.callback(
    Output('wordclouds','src'),
    [Input('num_cluster_dropwdown', 'value'),
     Input('cluster_alg_dropwdown', 'value'),
     Input('viz_dim_dropwdown', 'value'),
     Input('dim_reduction_dropwdown', 'value')])
def update_image_src(num_clusters, clust_alg, viz_dim, dim_red):
    uri = JP(image_directory, images_basename + '_{}_{}_{}.png'.format(num_clusters, clust_alg, dim_red))
    uri_base64 = base64.b64encode(open(uri, 'rb').read()).decode('ascii')
    uri_base64 = 'data:image/png;base64,{}'.format(uri_base64)
    return uri_base64


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)

exit()
