import pandas as pd
from datetime import date, timedelta
from sqlalchemy import create_engine
import plotly
import random
import plotly.graph_objs as go
#import pymysql
from ta.trend import MACD
import myconfig
import yfinance as yf
### import mysql.connector
from dash import Dash, Input, Output, dcc, html


sqlEngine       = create_engine(myconfig.connection_str, pool_recycle=3600)

dbConnection    = sqlEngine.connect()

# Override Yahoo Finance 



'''
conn = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="stock"
)
'''
st = "AAPL"
##query = "SELECT * FROM overview where Symbol = %s "
query = '''
select Ov1.*,
Round(SQRT(22.5 * Ov1.EPS * Ov1.BookValue),2) Graham_Number,
Round(SQRT(22.5 * Ov1.EPS * Ov1.BookValue) - Ov1.`52WeekLow`,2) Value_Stocks
from overview Ov1
Inner Join 
(select Symbol, Max(LatestQuarter) Max_Quarter from overview Group By Symbol) Max_Quarter_Qry
On Max_Quarter_Qry.Max_Quarter = Ov1.LatestQuarter and Max_Quarter_Qry.Symbol = Ov1.Symbol '''
query_sp = "SELECT * FROM sandp"
query_history = "SELECT * FROM history"
##df = pd.read_sql(query, dbConnection, params=(st,))
df = pd.read_sql(query, dbConnection)
df_sp = pd.read_sql(query_sp, dbConnection)
#df_history = pd.read_sql(query_history, dbConnection)
#print(df)
dashboardData = (df_sp) 
overviewData = (df)
#history_data = (df_history.assign(Date=lambda data: pd.to_datetime(data["Date"], format="%Y-%m-%d")).sort_values(by="Date")) 
data = (
    pd.read_csv("E:/Users/mecostantino/OneDrive - Pellissippi State Community College/Desktop/ETSU School/CSCI 5050 Decision Support Systems/Decision SUpport Systems Final Project/materials-python-dash/avocado_analytics_3/avocado.csv")
    .assign(Date=lambda data: pd.to_datetime(data["Date"], format="%Y-%m-%d"))
    .sort_values(by="Date")
)
datepickerdata = pd.read_sql(query, dbConnection)



def datepickerfunc(datedata):
    datepickerdata = datedata

def generate_query(chart_symbol, start_date, end_date):
    sqlEngine       = create_engine(myconfig.connection_str, pool_recycle=3600)

    dbConnection    = sqlEngine.connect()

    query = "SELECT  * FROM history where Symbol = %s "
    df_history  = pd.read_sql(query, dbConnection, params=(chart_symbol,))
    history_data = (df_history.assign(Date=lambda data: pd.to_datetime(data["Date"], format="%Y-%m-%d")).sort_values(by="Date")) 
 
    return history_data.query(
            "Symbol == @chart_symbol"
            " and Date >= @start_date and Date <= @end_date")

def macd_graph(macd_symbol,start_d,end_d):
    yf.pdr_override()

# Create input field for our desired stock 

# Retrieve stock data frame (df) from yfinance API at an interval of 1m 
   # macd_df = yf.download(tickers='AAPL',period='1y',interval='1d')
    macd_df = yf.download(tickers=macd_symbol,start = start_d, end = end_d, interval='1d')

#macd_df['MA5'] = macd_df['Close'].rolling(window=5).mean()
#macd_df['MA20'] = macd_df['Close'].rolling(window=20).mean()

#print(macd_df)
# MACD
    macd = MACD(close=macd_df['Close'], 
                window_slow=26,
                window_fast=12, 
                window_sign=9)
    fig = go.Figure()
    colorsM = ['green' if val >= 0 
            else 'red' for val in macd.macd_diff()]
    fig.add_trace(go.Bar(x=macd_df.index, 
                        y=macd.macd_diff(),
                        marker_color=colorsM,
                         name="Historgram"
                        ))
    fig.add_trace(go.Scatter(x=macd_df.index,
                         y=macd.macd(),
                         line=dict(color='black', width=2),
                          name="MACD"
                        ))
    fig.add_trace(go.Scatter(x=macd_df.index,
                         y=macd.macd_signal(),
                         line=dict(color='blue', width=1),
                         name="Signal"
                        ))
    fig.update_layout(title='MACD for ' + macd_symbol)

    return fig
    #fig.show()

def generate_table(qry_symbol, max_rows=26):
    data3 = {'Cap' : ['A', 'B', 'C', ], 'non-Cap' : ['a','b','c', ]}
    df = pd.DataFrame(data3)
    #print(df)
    filtered_o_data = overviewData.query(
        "Symbol == @qry_symbol"
    )
    #idx = filtered_o_data.groupby('Symbol')['PERatio'].idxmax()
    #max_overview = filtered_o_data.loc[filtered_o_data.groupby('Symbol')['LatestQuarter'].transform(max) == filtered_o_data['LatestQuarter']]
    #max_overview = filtered_o_data.loc[idx]
    #print(filtered_o_data)
    table_overview = filtered_o_data[['Symbol','Exchange', 'LatestQuarter', 'PERatio', 'PEGRatio', 'EPS','PriceToBookRatio', 'Beta', 'Graham_Number', 'Value_Stocks', '50DayMovingAverage','200DayMovingAverage']].stack()
    #table_overview.reset_index()
    todf = table_overview.reset_index(level=1)
    #print(table_overview.reset_index(level=1))
    #i need to iterate through rows instead of columns now
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns]) ] +
        # Body
        [html.Tr([
            html.Td(todf.iloc[i][col]) for col in todf.columns
        ]) for i in range(min(len(todf), max_rows))]
    )

#symbols = dashboardData["Symbol"] + "-" + dashboardData["Security"]

#symbols.sort_values().unique()

symbols = dashboardData["Symbol"].sort_values().unique()

regions = data["region"].sort_values().unique() 
avocado_types = data["type"].sort_values().unique()

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Stock Market Analytics: Understand Stocks!"



app.layout = html.Div(
    children=[
        html.Div(
            children=[
               # html.P(children="🥑", className="header-emoji"),
                #dcc.Input(id='num', type='number', debounce=True, min=2, step=1),
                html.H1(
                    children="Stock Analysis", className="header-title"
                ),
                html.P(
                    children=(
                        "Analyze the behavior of stock prices and the number"
                        " of stocks sold"
                    ),
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Symbol", className="menu-title"),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[
                                {"label": symbol, "value": symbol}
                                for symbol in symbols
                            ],
                            value="AAPL",
                            clearable=False,
                            className="dropdown",
 
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Type", className="menu-title"),
                        dcc.Dropdown(
                            id="type-filter",
                            options=[
                                {
                                    "label": avocado_type.title(),
                                    "value": avocado_type,
                                }
                                for avocado_type in avocado_types
                            ],
                            value="organic",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range", className="menu-title"
                        ),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=date(1980,1,1),
                            max_date_allowed=date.today(),
                            start_date=date.today() - timedelta(days=365),
                            end_date=date.today(),
                        ),
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[ html.Div(children=[
                html.Div( 
                    children=dcc.Graph(
                        id="price-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card-row",
                ),
                html.Div(children=[
                        html.P(
                        # generate_table(df)
                        id="stock-table",
                        ),
                        html.P(id="profit-value"), 
                ],
                className="right_header2 card-row"),
                html.Div(
                    children=dcc.Graph(
                        id="volume-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card-row",
                ),
            ], className="cards-side"),
                 html.Div(
                    children=dcc.Graph(
                        id="macd-chart",
                        config={"displayModeBar": False},
                       # figure=macd_graph(),
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),
    ]
)

dbConnection.close()



@app.callback(
    Output("price-chart", "figure"),
    Output("profit-value", "children"),
    Output("profit-value", "style"),
    Output("volume-chart", "figure"),
    Output("macd-chart", "figure"),
    Input("region-filter", "value"),
    Input("type-filter", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_charts(symbol, avocado_type, start_date, end_date):
    filtered_data = generate_query(symbol, start_date, end_date)
    begin_price = filtered_data["Adj_Close"].iloc[0]
    end_price = filtered_data["Adj_Close"].iloc[-1] 
    percent_return = round(((end_price - begin_price)/begin_price)*100,2)
    print(percent_return)
    
    if percent_return > 0:
        dynamic_style = {'color': 'green'}
    else:
        dynamic_style = {'color': 'red'}

    profit_return_value = f"Profit: {percent_return}%"
    macd_chart = macd_graph(symbol, start_date, end_date)
    price_chart_figure = {
        "data": [
            {
                "x": filtered_data["Date"],
                "y": filtered_data["Adj_Close"],
                "type": "lines",
                "hovertemplate": "$%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Adjusted Close History of " + symbol,
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True},
            "yaxis": {"tickprefix": "$", "fixedrange": True},
            "colorway": ["#17B897"],
        },
    }

    volume_chart_figure = {
        "data": [
            {
                "x": filtered_data["Date"],
                "y": filtered_data["Volume"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": symbol + " Volume", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }
    return price_chart_figure, profit_return_value, dynamic_style, volume_chart_figure , macd_chart 

@app.callback(
    Output("stock-table", "children"),
    Input("region-filter", "value")

)

def update_table(symbol): 
    
    return generate_table(symbol)







if __name__ == "__main__":
    app.run_server(debug=True)
