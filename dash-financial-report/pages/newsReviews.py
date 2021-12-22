from dash import html #import dash_html_components as html
from utils import Header
from dash import dcc #import dash_core_components as dcc
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pathlib

from newsapi import NewsApiClient
import pandas as pd

import datetime as dt
import pandas_datareader.data as web

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

#NewsAPI Key: 76982d41f138438593b035d85127b2a2

api = NewsApiClient(api_key='76982d41f138438593b035d85127b2a2')

news_SPI = api.get_everything(q='Swiss Market')
articles_SPI = news_SPI['articles']

collect_articles_SPI = []
for x, y in enumerate(articles_SPI):
    collect_articles_SPI.append(y["title"])

news_world = api.get_everything(q='Financial Market')
articles_world = news_world['articles']

collect_articles_world = []
for x, y in enumerate(articles_world):
    collect_articles_world.append(y["title"])
    
start = dt.datetime(2009,1,1)
end = dt.datetime(2021,10,1)
# S&P500 (^GSPC)
sp500 = web.DataReader('^GSPC', 'yahoo', start=start, end=end)
sp500['Date'] = sp500.index.date

df_spi_value = pd.read_csv(DATA_PATH.joinpath("SPI_value.csv"))


def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 6
            html.Div(
                [
                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("News in the Swiss Market", className="subtitle padded"),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.Li(
                                                collect_articles_SPI[0]
                                            ),
                                            html.Li(
                                                collect_articles_SPI[1]
                                            ),
                                            html.Li(
                                                collect_articles_SPI[2]
                                            ),
                                            html.Li(
                                                collect_articles_SPI[3]
                                            ),
                                            html.Li(
                                                collect_articles_SPI[4]
                                            ),
                                            html.Li(
                                                collect_articles_SPI[5]
                                            ),
                                            html.Li(
                                                collect_articles_SPI[6]
                                            ),
                                            html.Li(
                                                collect_articles_SPI[7]
                                            ),
                                            html.Li(
                                                collect_articles_SPI[8]
                                            ),
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                className="row",
                            ),
                            html.Div(
                                [
                                    html.H6("News in the Global Financial Market", className="subtitle padded"),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.Li(
                                                collect_articles_world[0]
                                            ),
                                            html.Li(
                                                collect_articles_world[1]
                                            ),
                                            html.Li(
                                                collect_articles_world[2]
                                            ),
                                            html.Li(
                                                collect_articles_world[3]
                                            ),
                                            html.Li(
                                                collect_articles_world[4]
                                            ),
                                            html.Li(
                                                collect_articles_world[5]
                                            ),
                                            html.Li(
                                                collect_articles_world[6]
                                            ),
                                            html.Li(
                                                collect_articles_world[7]
                                            ),
                                            html.Li(
                                                collect_articles_world[8]
                                            ),
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                className="row",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                        html.H6(
                                            "Evolution of S&P500 Index",
                                            className="subtitle padded",
                                        ),
                                        dcc.Graph(
                                            id="graph-sp500",
                                            figure={
                                                "data": [
                                                    go.Scatter(
                                                        x = sp500.index.date,
                                                        y = sp500['Close'],
                                                        line={"color": "#97151c"},
                                                        mode="lines",
                                                        name="S&P500 Index (USD)",
                                                    )
                                                ],
                                                "layout": go.Layout(
                                                    autosize=True,
                                                    title="",
                                                    font={"family": "Raleway", "size": 10},
                                                    height=200,
                                                    width=340,
                                                    hovermode="closest",
                                                    legend={
                                                        "x": -0.0277108433735,
                                                        "y": -0.142606516291,
                                                        "orientation": "h",
                                                    },
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
                                    className="six columns",
                                ),                                   
                                    html.Div(
                                        [
                                        html.H6(
                                            "Evolution of Swiss Performance Index",
                                            className="subtitle padded",
                                        ),
                                        dcc.Graph(
                                            id="graph-spi",
                                            figure={
                                                "data": [
                                                    go.Scatter(
                                                        x = df_spi_value['Date'],
                                                        y = df_spi_value['SPI INDEX'],
                                                        line={"color": "#97151c"},
                                                        mode="lines",
                                                        name="Swiss Performance Index  (CHF)",
                                                    )
                                                ],
                                                "layout": go.Layout(
                                                    autosize=True,
                                                    title="",
                                                    font={"family": "Raleway", "size": 10},
                                                    height=200,
                                                    width=340,
                                                    hovermode="closest",
                                                    legend={
                                                        "x": -0.0277108433735,
                                                        "y": -0.142606516291,
                                                        "orientation": "h",
                                                    },
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
                                    className="six columns",
                                ),
                            ],
                            className="row ",
                            ),
                            html.Div(
                                [
                                    html.H6("Disclaimer", className="subtitle padded"),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.P(
                                                "This publication is solely meant to be part of an informal project of the course Quantitative Asset & Risk Management II given in Autumn 2021 of the \
                                                University of Lausanne and is no way an offer or invitation to buy or sell investment or any other products. The information and opinions reflected in this document originate from reliable sources;\
                                                nonetheless, we refuse and dismiss any contractual or implicit liability for incorrect or incomplete information, judgments or estimations. All information, opinions and numbers\
                                                can be subject to change at any moment without advance notice."
                                            ),
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                className="row",
                            ),
                        ],
                        className="row ",
                    )
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
