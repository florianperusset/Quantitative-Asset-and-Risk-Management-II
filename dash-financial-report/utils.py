from dash import dcc #import dash_core_components as dcc
from dash import html #import dash_html_components as html


def Header(app):
    return html.Div([get_header(app), html.Br([]), get_menu()])


def get_header(app):
    header = html.Div(
        [
            html.Div(
                [
                    html.A(
                        html.Img(
                            src=app.get_asset_url("logo_qarm_capital.png"),
                            className="logo",
                        ),
                        href="https://www.unil.ch/hec/en/home.html",
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H5("Style Rotation on Swiss Long-Only Equity Factors Funds")],
                        className="seven columns main-title",
                    ),
                    html.Div(
                        [
                            dcc.Link(
                                "Full View",
                                href="/dash-financial-report/full-view",
                                className="full-view-link",
                            )
                        ],
                        className="five columns",
                    ),
                ],
                className="twelve columns",
                style={"padding-left": "0"},
            ),
        ],
        className="row",
    )
    return header


def get_menu():
    menu = html.Div(
        [
            dcc.Link(
                "Overview",
                href="/dash-financial-report/overview",
                className="tab first",
            ),
            dcc.Link(
                "Defensive Portfolio",
                href="/dash-financial-report/portfolio-defensive",
                className="tab",
            ),
            dcc.Link(
                "Balanced Portfolio",
                href="/dash-financial-report/portfolio-equilibrium",
                className="tab",
            ),
            dcc.Link(
                "Dynamic Portfolio", 
                href="/dash-financial-report/portfolio-aggressive", 
                className="tab",
            ),
            dcc.Link(
                "News & Reviews",
                href="/dash-financial-report/news-and-reviews",
                className="tab",
            ),
        ],
        className="row all-tabs",
    )
    return menu


def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table
