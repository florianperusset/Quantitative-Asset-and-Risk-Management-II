from dash import dcc #import dash_core_components as dcc
from dash import html #import dash_html_components as html
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from utils import Header, make_dash_table 

import pandas as pd
import pathlib

import datetime as dt
import pandas_datareader.data as web

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

df_fund_facts = pd.read_csv(DATA_PATH.joinpath("df_fund_facts.csv"))
df_returns_all_ptf = pd.read_csv(DATA_PATH.joinpath("returns_all_ptf.csv"))
df_perf_all_ptf = pd.read_csv(DATA_PATH.joinpath("perf_all_basis.csv"))

def create_layout(app):
    # Page layouts
    return html.Div(
        [
            html.Div([Header(app)]), 
            # page 1
            html.Div(
                [
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Funds Profile"),
                                    html.Br([]),
                                    html.P(
                                        "\
                                            Our client is a large Swiss pension fund who has a substantial allocation to Swiss equities. He is a firm believer of risk premia and is fully convinced by the long-term benefits of tilting his portfolio to reap the benefits of well-known risk premia.\
                                            With no particular view on which risk premia is best suited for him, he wants to go for a diversified approach. He is nevertheless concerned by the time-varying nature of factor returns and fears of being unable to cope with a too long period of underperformance of one given factor.\
                                            He is therefore thinking about the potentials of adjusting his exposure to the various risk premia over time and make his portfolio more dynamic.\
                                            He is willing to give a mandate for managing a dynamic long-only portfolio of risk premia on the Swiss market. Tracking error is also a concern for him.",
                                        style={"color": "#ffffff"},
                                        className="row",
                                    ),
                                    html.P(
                                        "Following our mandate, we will propose three types of funds (i.e. defensive, balanced and dynamic) with various risk-returns trade-off and results. Nonetheless, our objective\
                                        is to create capital growth over the long-term with actively managed funds exposed to the Swiss market.",
                                        style={"color": "#ffffff"},
                                        className="row",
                                    ),                                        
                                    html.P(
                                        "Fund Managers: Sebastien Gorgoni, Florian Perusset, Florian Vogt",
                                        style={"color": "#ffffff"},
                                        className="row",
                                    ),
                                ],
                                className="product",
                            )
                        ],
                        className="row",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Fund Facts"], className="subtitle padded"
                                    ),
                                    html.Table(make_dash_table(df_fund_facts)),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Portfolio Performances"], className="subtitle padded"
                                    ),
                                    html.Table(make_dash_table(df_perf_all_ptf)),
                                ],
                                className="row",
                            ),
                        ],
                        className="row",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("Cumulative Performances", className="subtitle padded"),
                                      dcc.Dropdown(
                                        id='portfolio-dropdown-all',
                                        options=[
                                            {'label': 'Defensive Portfolio', 'value': 'defensive'},
                                            {'label': 'Balanced Portfolio', 'value': 'balanced'},
                                            {'label': 'Dynamic Portfolio', 'value': 'dynamic'},
                                        ],
                                        value='defensive'
                                    ),
                                    html.Div(id='dd-output-container'),                               
                                    dcc.Graph(
                                        id="graph-portfolio-all",
                                        figure={
                                            "data": [
                                                go.Scatter( 
                                                    x = df_returns_all_ptf["Date"],
                                                    y = df_returns_all_ptf["Balanced Portfolio"],
                                                    line={"color": "#97151c"},
                                                    mode="lines",
                                                    name="Balanced Portfolio",
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
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="twelve columns", 
                            )
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
