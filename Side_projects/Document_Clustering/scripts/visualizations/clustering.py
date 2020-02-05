
import os
import sys
import base64
from io import BytesIO

import pickle
import pandas as pd
from os.path import join as JP

import dash
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import dash_core_components as dcc
import dash_html_components as html
from plotly.tools import mpl_to_plotly
from plotly.subplots import make_subplots
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



''' ---------------------------------- DATA ---------------------------------- '''

images_basename = 'wordcloud_clusters'
logo = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/BASF-Logo_bw.svg/1280px-BASF-Logo_bw.svg.png'

with open(JP(paths['results'], 'clustering.pkl'), 'rb') as f:
    results = pickle.load(f)

''' ---------------------------------- HP ---------------------------------- '''

VIZ_DIMENSIONS = [2,3]
K_VALUES = list(results.keys())
WORDS_PER_VALUE = list(range(3,10))
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

    ], className = 'row', style={
        'backgroundColor':'green', 
        'text-align':'center', 
        'padding':'20px',
        'font-family':'verdana',
        'font-size':'150%',
        'color':'white'}),    
    
    # BODY
    html.Div([

        # FIRST PART (SCATTER + BARPLOTS)
        html.Div([

            # LEFT-HAND CHART --> SCATTER PLOT
            html.Div([
                
                # LEFT TOP HALF --> 4 DROPDOWNS
                html.Div([
                    
                    # LEFT TOP HALF LEFT HALF --> FIRST 2 DROPDOWN
                    html.Div([
                        
                        # DROPDOWN: Number of clusters ?
                        html.Div([
                            # Title
                            html.Div(
                                [html.P('Number of Clusters')
                            ], id='dropdown_1_text', className='six columns'),
                            # Dropdown
                            html.Div([
                                dcc.Dropdown(
                                id='num_cluster_dropwdown',
                                options=[{'label': i, 'value': i} for i in K_VALUES],
                                value=K_VALUES[0])
                            ], id='dropdown_1_box', className='six columns')

                        ], id='dropdown_1', className='row', style={'padding-bottom':'5px'}), 

                        # DROPDOWN: Clustering method ? 
                        html.Div([
                            # Title
                            html.Div([
                                html.P('Clustering method')
                                ], id='dropdown_2_text', className='six columns'),
                            # Dropdown
                            html.Div([
                                dcc.Dropdown(
                                id='cluster_alg_dropwdown',
                                options=[{'label': i, 'value': i} for i in CLUST_METHODS],
                                value=CLUST_METHODS[1])
                            ], id='dropdown_2_box', className='six columns')

                        ], id='dropdown_2',  className='row', style={'padding-bottom':'5px'})  

                    ], id='umap_dropdowns_1_and_2', className='six columns', style={'margin-top':'5%', 'padding-left':'5%'}),

                    # LEFT TOP HALF RIGHT HALF --> FIRST 2 DROPDOWN
                    html.Div([
                            
                        # DROPDOWN: DIMESIONS TO VISUALIZE ? 
                        html.Div([
                            
                            # Title
                            html.Div([
                                html.P('Plot Dimensions')
                            ], className='six columns'),
                            
                            # Dropdown
                            html.Div([
                                dcc.Dropdown(
                                id='viz_dim_dropwdown',
                                options=[{'label': i, 'value': i} for i in VIZ_DIMENSIONS],
                                value=3)
                            ], className='six columns')

                        ], className='row', style={'padding-bottom':'5px'}), 

                        # DROPDOWN: DIM REDUCTOR? s
                        html.Div([
                            
                            # Title
                            html.Div([
                                html.P('Dim Reductor')
                            ], className='six columns'),
                            
                            # Dropdown
                            html.Div([
                                dcc.Dropdown(
                                id='dim_reduction_dropwdown',
                                options=[{'label': i, 'value': i} for i in DIM_REDUCTORS],
                                value=DIM_REDUCTORS[0])
                            ], className='six columns')

                        ], className='row', style={'padding-bottom':'5px'}), 

                    ], id='umap_dropdowns_3_and_4', className='six columns', style={'margin-top':'5%', 'padding-left':'5%'})

                ], id='umap_dropdown', className = 'row'),

                # LEFT BOTTOM HALF --> SCATTER PLOT
                html.Div([
                    dcc.Graph(id='umap')
                ], id='umap_graph', className = 'row'),

            ], id='scatter_container', className = 'six columns'),

            # RIGHT-HAND HALF --> BARPLOTS
            html.Div([

                # DROPDOWN: WORDS PER WORDCLOUD
                html.Div([
                    
                    # TITLE
                    html.Div(
                        [html.P('Number of Words per Cluster')
                    ], id='words_per_wordcloud_text', className='six columns'),
                    
                    # DROPDOWN
                    html.Div([
                        dcc.Dropdown(
                        id='words_per_barplot_dropdown',
                        options=[{'label': i, 'value': i} for i in WORDS_PER_VALUE],
                        value=WORDS_PER_VALUE[3])
                    ], id='words_per_wordcloud_box', className='six columns')

                ], id='barplot_dropdown', className='row', style={'margin':'40px'}),

                html.Div([
                    dcc.Graph(id='barplots')
                ], id='barplot_graph', className = 'row'),

            ], id='barplot_container', className='six columns'),

        ], id='scatter_barplot_container', className='row'),

        # SECOND PART WORDCLOUD
        html.Div([
            html.Div([
                html.Div([html.Img(id='wordclouds', src='')])
            ], className='row')

        ], id='wordcloud_frame', className = 'row',
            style={'text-align':'center', 'padding':'20px'})

    ], className = 'row')
])

@app.callback(
    Output('umap','figure'), 
    [Input('num_cluster_dropwdown', 'value'),
     Input('cluster_alg_dropwdown', 'value'),
     Input('viz_dim_dropwdown', 'value'),
     Input('dim_reduction_dropwdown', 'value')])
def update_umap(num_clusters, clust_alg, viz_dim, dim_red):
    # Filter based on dropdown values
    d = results[num_clusters][clust_alg][dim_red]['scatter']   
    # Convert Cluster to String to be treated as a Category
    d['cluster'] = d['cluster'].astype(str)
    # Different plot depending on the desired dimensions to display
    if viz_dim == 3:
        fig = px.scatter_3d(d, x='d1', y='d2', z='d3', color='cluster')
    else:
        fig = px.scatter(d, x='d1', y='d2', color='cluster')
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(title_text='Document embeddings', title_x=0.5, showlegend=False)
    return fig 


@app.callback(
    Output('barplots', 'figure'),
    [Input('num_cluster_dropwdown', 'value'),
     Input('cluster_alg_dropwdown', 'value'),
     Input('dim_reduction_dropwdown', 'value'),
     Input('words_per_barplot_dropdown', 'value')])
def update_barplot(num_clusters, clust_alg, dim_red, words_per_value):
    d = results[num_clusters][clust_alg][dim_red]['wordclouds']
    # Create subplots for each number of clusters option
    fig = make_subplots(rows=num_clusters, cols=1)
    # Populate each of the subplots, filtering by cluster, by score, and taking the N with smax scores
    for c, cluster in enumerate(d['cluster'].unique()):
        d_ = d[d['cluster'] == cluster].sort_values(by='score', ascending=False).head(words_per_value)
        fig.append_trace(go.Bar(x=d_['word'], y=d_['score'], orientation='v'), row=c+1, col=1)
    fig.update_layout(title_text='Word Importance')
    return fig


@app.callback(
    Output('wordclouds','src'),
    [Input('num_cluster_dropwdown', 'value'),
     Input('cluster_alg_dropwdown', 'value'),
     Input('dim_reduction_dropwdown', 'value')])
    #  Input('n_cols_dropwdown', 'value')
def update_wordclouds(num_clusters, clust_alg, dim_red): #, n_cols):
    # Filter data
    d = results[num_clusters][clust_alg][dim_red]['wordclouds'] 
    # Create the figure using pyplot
    fig = plot_centroids_as_wordclouds(d, n_cols=num_clusters, figsize=(10,5))
    # fig.savefig(JP(paths['images'], 'test_{}_{}_{}.png'.format(num_clusters, clust_alg, dim_red)))
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    # Embed the result in the html output.
    return 'data:image/png;base64,{}'.format(base64.b64encode(buf.read()).decode("ascii"))

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)

exit()

# BACKUP CODE

# When wordcloud were not computed on the fly
# @app.callback(
#     Output('wordclouds','src'),
#     [Input('num_cluster_dropwdown', 'value'),
#      Input('cluster_alg_dropwdown', 'value'),
#      Input('dim_reduction_dropwdown', 'value')])
# def update_wordclouds(num_clusters, clust_alg, dim_red):
#     uri = JP(image_directory, images_basename + '_{}_{}_{}.png'.format(num_clusters, clust_alg, dim_red))
#     uri_base64 = base64.b64encode(open(uri, 'rb').read()).decode('ascii')
#     uri_base64 = 'data:image/png;base64,{}'.format(uri_base64)
#     return uri_base64
