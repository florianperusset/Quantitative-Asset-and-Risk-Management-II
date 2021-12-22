# -*- coding: utf-8 -*-
import dash
from dash import dcc #import dash_core_components as dcc
from dash import html #import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go

from pages import (
    overview,
    portfolioDefensive,
    portfolioEquilibrium, 
    portfolioAggressive, 
    newsReviews,
)

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "Financial Report"
server = app.server

# Describe the layout/ UI of the app
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

# Update page
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/dash-financial-report/portfolio-defensive":
        return portfolioDefensive.create_layout(app)
    elif pathname == "/dash-financial-report/portfolio-equilibrium":
        return portfolioEquilibrium.create_layout(app)
    elif pathname == "/dash-financial-report/portfolio-aggressive": 
        return portfolioAggressive.create_layout(app)
    elif pathname == "/dash-financial-report/news-and-reviews":
        return newsReviews.create_layout(app)
    elif pathname == "/dash-financial-report/full-view":
        return (
            overview.create_layout(app),
            portfolioDefensive.create_layout(app),
            portfolioEquilibrium.create_layout(app),
            portfolioAggressive.create_layout(app),
            newsReviews.create_layout(app),
        )
    else:
        return overview.create_layout(app)
    
@app.callback(
    Output(component_id='graph-portfolio-all',component_property='figure'),
    [Input(component_id='portfolio-dropdown-all', component_property='value')]
) 

def update_overview(value):
    
    df_returns_all_ptf = pd.read_csv("data/returns_all_ptf.csv")
    
    if value == 'defensive':
        xx = df_returns_all_ptf['Defensive Portfolio']
        name_plot = 'Defensive Portfolio'
        
    elif value == 'balanced':
        xx = df_returns_all_ptf['Balanced Portfolio']
        name_plot = 'Balanced Portfolio'
        
    elif value == 'dynamic':
        xx = df_returns_all_ptf['Dynamic Portfolio'] 
        name_plot = 'Dynamic Portfolio'
    
    return {
            "data": [
                go.Scatter( 
                    x= df_returns_all_ptf["Date"],
                    y= xx,
                    line={"color": "#97151c"},
                    mode="lines",
                    name=name_plot,
                ),
                go.Scatter( 
                    x= df_returns_all_ptf["Date"],
                    y= df_returns_all_ptf["Benchmark"],
                    line={"color": "#b5b5b5"},
                    mode="lines",
                    name='Benchmark',
                ),
            ],
            "layout": go.Layout(
                autosize=True,
                width=700,
                height=200,
                font={"family": "Raleway", "size": 10},
                margin={
                    "r": 30,
                    "t": 30,
                    "b": 30,
                    "l": 30,
                },
                showlegend=True,
                titlefont={
                    "family": "Raleway",
                    "size": 10,
                },
                xaxis={
                    "autorange": True,
                    "range": [
                        "2007-12-31",
                        "2018-03-06",
                    ],
                    "rangeselector": {
                        "buttons": [
                            {
                                "count": 1,
                                "label": "1Y",
                                "step": "year",
                                "stepmode": "backward",
                            },
                            {
                                "count": 3,
                                "label": "3Y",
                                "step": "year",
                                "stepmode": "backward",
                            },
                            {
                                "count": 5,
                                "label": "5Y",
                                "step": "year",
                            },
                            {
                                "count": 10,
                                "label": "10Y",
                                "step": "year",
                                "stepmode": "backward",
                            },
                            {
                                "label": "All",
                                "step": "all",
                            },
                        ]
                    },
                    "showline": True,
                    "type": "date",
                    "zeroline": False,
                },
                yaxis={
                    "autorange": True,
                    "range": [
                        18.6880162434,
                        278.431996757,
                    ],
                    "showline": True,
                    "type": "linear",
                    "zeroline": False,
                },
            ),
        }

@app.callback(
    Output(component_id='graph-portfolio-defensive',component_property='figure'),
    [Input(component_id='portfolio-dropdown-defensive', component_property='value')]
)

def update_map_defensive(value): 
    
    df_returns_ridge = pd.read_csv("data/returns_ridge.csv")
    
    if value == '100P':
        xx = df_returns_ridge["100% Portfolio"]
        name_plot = '100% Portfolio'
        
    elif value == '80P20B':
        xx = df_returns_ridge["80% Portfolio, 20% Benchmark"]
        name_plot = '80% Portfolio, 20% Benchmark'
           
    return {
            "data": [
                go.Scatter( 
                    x= df_returns_ridge["Date"],
                    y= xx,
                    line={"color": "#97151c"},
                    mode="lines",
                    name='Portfolio',
                ),
                go.Scatter( 
                    x= df_returns_ridge["Date"],
                    y= df_returns_ridge["Benchmark"],
                    line={"color": "#b5b5b5"},
                    mode="lines",
                    name='Benchmark',
                ),
            ],
            "layout": go.Layout(
                autosize=True,
                width=700,
                height=200,
                font={"family": "Raleway", "size": 10},
                margin={
                    "r": 30,
                    "t": 30,
                    "b": 30,
                    "l": 30,
                },
                showlegend=True,
                titlefont={
                    "family": "Raleway",
                    "size": 10,
                },
                xaxis={
                    "autorange": True,
                    "range": [
                        "2007-12-31",
                        "2018-03-06",
                    ],
                    "rangeselector": {
                        "buttons": [
                            {
                                "count": 1,
                                "label": "1Y",
                                "step": "year",
                                "stepmode": "backward",
                            },
                            {
                                "count": 3,
                                "label": "3Y",
                                "step": "year",
                                "stepmode": "backward",
                            },
                            {
                                "count": 5,
                                "label": "5Y",
                                "step": "year",
                            },
                            {
                                "count": 10,
                                "label": "10Y",
                                "step": "year",
                                "stepmode": "backward",
                            },
                            {
                                "label": "All",
                                "step": "all",
                            },
                        ]
                    },
                    "showline": True,
                    "type": "date",
                    "zeroline": False,
                },
                yaxis={
                    "autorange": True,
                    "range": [
                        18.6880162434,
                        278.431996757,
                    ],
                    "showline": True,
                    "type": "linear",
                    "zeroline": False,
                },
            ),
        }

@app.callback(
    Output(component_id='graph-portfolio-balanced',component_property='figure'),
    [Input(component_id='portfolio-dropdown-balanced', component_property='value')]
)

def update_map_balanced(value): 
    
    df_returns_parametrics = pd.read_csv("data/returns_parametrics.csv")
    
    if value == '100P':
        xx = df_returns_parametrics["100% Portfolio"]
        name_plot = '100% Portfolio'
        
    elif value == '80P20B':
        xx = df_returns_parametrics["80% Portfolio, 20% Benchmark"]
        name_plot = '80% Portfolio, 20% Benchmark'
           
    return {
            "data": [
                go.Scatter( 
                    x= df_returns_parametrics["Date"],
                    y= xx,
                    line={"color": "#97151c"},
                    mode="lines",
                    name='Portfolio',
                ),
                go.Scatter( 
                    x= df_returns_parametrics["Date"],
                    y= df_returns_parametrics["Benchmark"],
                    line={"color": "#b5b5b5"},
                    mode="lines",
                    name='Benchmark',
                ),
            ],
            "layout": go.Layout(
                autosize=True,
                width=700,
                height=200,
                font={"family": "Raleway", "size": 10},
                margin={
                    "r": 30,
                    "t": 30,
                    "b": 30,
                    "l": 30,
                },
                showlegend=True,
                titlefont={
                    "family": "Raleway",
                    "size": 10,
                },
                xaxis={
                    "autorange": True,
                    "range": [
                        "2007-12-31",
                        "2018-03-06",
                    ],
                    "rangeselector": {
                        "buttons": [
                            {
                                "count": 1,
                                "label": "1Y",
                                "step": "year",
                                "stepmode": "backward",
                            },
                            {
                                "count": 3,
                                "label": "3Y",
                                "step": "year",
                                "stepmode": "backward",
                            },
                            {
                                "count": 5,
                                "label": "5Y",
                                "step": "year",
                            },
                            {
                                "count": 10,
                                "label": "10Y",
                                "step": "year",
                                "stepmode": "backward",
                            },
                            {
                                "label": "All",
                                "step": "all",
                            },
                        ]
                    },
                    "showline": True,
                    "type": "date",
                    "zeroline": False,
                },
                yaxis={
                    "autorange": True,
                    "range": [
                        18.6880162434,
                        278.431996757,
                    ],
                    "showline": True,
                    "type": "linear",
                    "zeroline": False,
                },
            ),
        }

@app.callback(
    Output(component_id='graph-portfolio-dynamic',component_property='figure'),
    [Input(component_id='portfolio-dropdown-dynamic', component_property='value')]
)

def update_map_dynamic(value): 
    
    df_returns_mom_factors = pd.read_csv("data/returns_mom_factors.csv")
    
    if value == '100P':
        xx = df_returns_mom_factors["100% Portfolio"]
        name_plot = '100% Portfolio'
        
    elif value == '80P20B':
        xx = df_returns_mom_factors["80% Portfolio, 20% Benchmark"]
        name_plot = '80% Portfolio, 20% Benchmark'
           
    return {
            "data": [
                go.Scatter( 
                    x= df_returns_mom_factors["Date"],
                    y= xx,
                    line={"color": "#97151c"},
                    mode="lines",
                    name='Portfolio',
                ),
                go.Scatter( 
                    x= df_returns_mom_factors["Date"],
                    y= df_returns_mom_factors["Benchmark"],
                    line={"color": "#b5b5b5"},
                    mode="lines",
                    name='Benchmark',
                ),
            ],
            "layout": go.Layout(
                autosize=True,
                width=700,
                height=200,
                font={"family": "Raleway", "size": 10},
                margin={
                    "r": 30,
                    "t": 30,
                    "b": 30,
                    "l": 30,
                },
                showlegend=True,
                titlefont={
                    "family": "Raleway",
                    "size": 10,
                },
                xaxis={
                    "autorange": True,
                    "range": [
                        "2007-12-31",
                        "2018-03-06",
                    ],
                    "rangeselector": {
                        "buttons": [
                            {
                                "count": 1,
                                "label": "1Y",
                                "step": "year",
                                "stepmode": "backward",
                            },
                            {
                                "count": 3,
                                "label": "3Y",
                                "step": "year",
                                "stepmode": "backward", 
                            },
                            {
                                "count": 5,
                                "label": "5Y",
                                "step": "year",
                            },
                            {
                                "count": 10,
                                "label": "10Y",
                                "step": "year",
                                "stepmode": "backward",
                            },
                            {
                                "label": "All",
                                "step": "all",
                            },
                        ]
                    },
                    "showline": True,
                    "type": "date",
                    "zeroline": False,
                },
                yaxis={
                    "autorange": True,
                    "range": [
                        18.6880162434,
                        278.431996757,
                    ],
                    "showline": True,
                    "type": "linear",
                    "zeroline": False,
                },
            ),
        }



if __name__ == "__main__":
    app.run_server(debug=True)

# if __name__ == '__main__':
#     app.server.run(port=8000, host='127.0.0.1', debug=True)
