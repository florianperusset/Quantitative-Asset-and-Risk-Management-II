from dash import dcc #import dash_core_components as dcc
from dash import html #import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from utils import Header, make_dash_table
import pandas as pd
import pathlib


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

df_perf_ptf = pd.read_csv(DATA_PATH.joinpath("perf_parametrics_basis.csv"))
df_weights_factors = pd.read_csv(DATA_PATH.joinpath("weights_parametrics_basis.csv")) 
df_returns_ptf = pd.read_csv(DATA_PATH.joinpath("returns_parametrics.csv"))
df_avg_returns = pd.read_csv(DATA_PATH.joinpath("avg_returns_parametrics.csv"))

def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 2 
            html.Div(
                [
                    html.Div( 
                        [
                            html.H5("Balanced Portfolio"),
                            html.Br([]),
                            html.P(
                                "\
                                    Our balanced portfolio aims to provide an investment solution with an optimal trade-off between risk and returns with a full exposure to Swiss equities with long-only positions.\
                                    This portfolio has the objective to moderately outperform the benchmark, thus keeping a moderate tracking-error.\
                                    Using advanced technics (i.e. parametric weights), we are able to appropriatly time our factor exposure using the VIX Index as our reference value of market sentiment to construct an optimal portfolio.\
                                    This portfolio is appropriate for investors which are willing to take a small amount of risk to gain outperformances.",
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
                    ),                   
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [  
                                    html.H6(
                                        ["Portfolio Performances"], className="subtitle padded"
                                    ),
                                    html.Table(make_dash_table(df_perf_ptf)),
                                ],
                                #className="row ",
                                className="eight columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        ["Average Factor Weights (%)"], className="subtitle padded"
                                    ),
                                    html.Table(make_dash_table(df_weights_factors)),     
                                ],
                                #className="row ",
                                className="four columns",
                            ),  
                        ],
                        className="row ",
                    ),
                    # New Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("Cumulative Performances", className="subtitle padded"),
                                     dcc.Dropdown(
                                        id='portfolio-dropdown-balanced',
                                        options=[
                                            {'label': '100% Portfolio', 'value': '100P'},
                                            {'label': '80% Portfolio, 20% Benchmark', 'value': '80P20B'},
                                        ],
                                        value='100P'
                                    ),
                                    html.Div(id='dd-output-container'),                               
                                    dcc.Graph(
                                        id="graph-portfolio-balanced",
                                        figure={
                                            "data": [
                                                go.Scatter( 
                                                    x = df_returns_ptf["Date"],
                                                    y = df_returns_ptf["100% Portfolio"],
                                                    line={"color": "#97151c"},
                                                    mode="lines",
                                                    name="100% Portfolio",
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
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "Average Annual Returns (%)" 
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.Div(
                                        [
                                            html.Table(
                                                make_dash_table(df_avg_returns),
                                                className="tiny-header",
                                            )
                                        ],
                                        style={"overflow-x": "auto"},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    html.Div(
                        [
                            html.H6(
                                "Risk Potential", className="subtitle padded"
                            ),
                            html.Img(
                                src=app.get_asset_url("risk_reward_3.png"), 
                                # style={'height':'110%', 'width':'110%'},
                                className="risk-reward",
                            ),
                        ],
                        className="row",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
