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
select Distinct Ov1.*,
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

def profit_return(kpi_filter,begin_date, end_date):
    overviewData[["PERatio", "PEGRatio",'PriceToBookRatio','Beta','Graham_Number']] = overviewData[["PERatio", "PEGRatio",'PriceToBookRatio','Beta','Graham_Number']].apply(pd.to_numeric, errors='coerce')
    #print(kpi_filter)
    match kpi_filter:

        case 'PERatio':
            #print("PERatio case statement")
            top_20_kpi = overviewData[['Symbol','PERatio']].sort_values(by ='PERatio').head(20).copy()
            top_20_kpi.columns = ["Symbol", "KPI"]
            #print(top_20_kpi)
        case 'PEGRatio':
            #print("PEGRatio case statement 2nd is working!")
            top_20_kpi = overviewData[['Symbol','PEGRatio']].sort_values(by ='PEGRatio',key=abs).head(20).copy()
            top_20_kpi.columns = ["Symbol", "KPI"]
        case 'PriceToBookRatio':
            #print("PEGRatio case statement 2nd is working!")
            top_20_kpi = overviewData[['Symbol','PriceToBookRatio']].sort_values(by ='PriceToBookRatio').head(20).copy()
            top_20_kpi.columns = ["Symbol", "KPI"]
        case 'Beta':
            #print("PEGRatio case statement 2nd is working!")
            top_20_kpi = overviewData[['Symbol','Beta']].sort_values(by ='Beta').head(20).copy()
            top_20_kpi.columns = ["Symbol", "KPI"]  
        case 'Graham_Number':
            #print("PEGRatio case statement 2nd is working!")
            top_20_kpi = overviewData[['Symbol','Value_Stocks']].sort_values(by ='Value_Stocks', ascending=False).tail(-1).head(20).copy()
            top_20_kpi.columns = ["Symbol", "KPI"]                   
        case _:

    
            sqlEngine       = create_engine(myconfig.connection_str, pool_recycle=3600)

            dbConnection    = sqlEngine.connect()

            query = '''Select Distinct max_info.Symbol, max_info.Date, max_info.Adj_Close_Max, min_info.Date, min_info.Adj_Close_Min,
                    Round(((max_info.Adj_Close_Max-min_info.Adj_Close_Min)/min_info.Adj_Close_Min)*100,2) Profits
                    from 
                        (
                    select distinct h1.Symbol, h1.Date, Round(h1.Adj_close,3) Adj_Close_Max from history h1 
                    JOIN # Derived Query 1
                        (Select Symbol, max(date) max_date from history where DATE(Date) between %(dstart)s  and %(dfinish)s Group By Symbol) max_qry
                    On h1.Symbol = max_qry.Symbol and h1.Date = max_qry.max_date ) max_info
                        JOIN (Select Distinct Symbol from history ) Symbol # Derived Query 2
                        On max_info.Symbol = Symbol.Symbol #Join Derived Query 1 and 2
                    JOIN (select distinct h2.Symbol, h2.Date, Round(h2.Adj_close,3) Adj_Close_Min from history h2 
                        JOIN # Derived Query 3
                        (Select Symbol, Date(min(date)) min_date from history where DATE(Date) between %(dstart)s  and %(dfinish)s Group By Symbol) min_qry
                            On h2.Symbol = min_qry.Symbol and h2.Date = min_qry.min_date) min_info
                        On Symbol.Symbol = min_info.Symbol #Join the Derivery Qry 2 and 3
                        Order By 6 Desc
                        Limit 20'''
            df_history  = pd.read_sql(query, dbConnection, params=({"dstart":begin_date,"dfinish":end_date}))
            #history_data = (df_history.assign(Date=lambda data: pd.to_datetime(data["Date"], format="%Y-%m-%d")).sort_values(by="Date")) 
            #print(df_history)

            bar_fig = go.Figure()
    
            bar_fig.add_trace(go.Bar(x=df_history['Symbol'], 
                        y=df_history['Profits'],
                        marker_color='green',
                        name="Profit"
                        ))
            dbConnection.close()

            bar_fig.update_yaxes(ticksuffix="%")
            bar_fig.update_layout(title=kpi_filter + " KPI")
            return bar_fig

    overview_fig = go.Figure()
    
    overview_fig.add_trace(go.Bar(x=top_20_kpi['Symbol'], 
                        y=top_20_kpi['KPI'],
                        marker_color='green',
                        name=kpi_filter
                        ))

    overview_fig.update_layout(title=kpi_filter + " KPI")
    #bar_fig.update_yaxes(ticksuffix="%")
    return overview_fig


def datepickerfunc(datedata):
    datepickerdata = datedata

def generate_query(chart_symbol, years_back, start_date, end_date):
    sqlEngine       = create_engine(myconfig.connection_str, pool_recycle=3600)

    dbConnection    = sqlEngine.connect()

    query = "SELECT  distinct * FROM history where Symbol = %(symbol)s "
    df_history  = pd.read_sql(query, dbConnection, params=({"symbol":chart_symbol}))
    history_data = (df_history.assign(Date=lambda data: pd.to_datetime(data["Date"], format="%Y-%m-%d")).sort_values(by="Date")) 
    #print(df_history)
    #print(history_data.groupby(pd.DatetimeIndex(history_data.Date).to_period('Y')).nth([0,-1]))
    #begin_end_year = history_data.groupby(pd.DatetimeIndex(history_data.Date).to_period('Y')).nth([0,-1])
    begin_year = history_data[['Symbol','Date','Adj_Close']].groupby(pd.DatetimeIndex(history_data.Date).to_period('Y')).nth([0]).copy()
    begin_year['Year'] = begin_year['Date'].dt.year
    end_year = history_data[['Symbol','Date','Adj_Close']].groupby(pd.DatetimeIndex(history_data.Date).to_period('Y')).nth([-1]).copy()
    end_year['Year'] = end_year['Date'].dt.year

    #begin_end_year['Adj_Close_diff'] = begin_end_year["Adj_Close"].diff()
    merged_dataframe = pd.merge_asof(begin_year,end_year,on="Year")

    merged_dataframe['Profits'] = ((merged_dataframe['Adj_Close_y'] - merged_dataframe['Adj_Close_x']) / merged_dataframe['Adj_Close_x'])*100

    #print(merged_dataframe)

    return_history = merged_dataframe.sort_values(by="Year",ascending=False).head(int(years_back))

    average_return = round(return_history['Profits'].mean(),2)

    #print(average_return)

    #print(return_history)

    annual_bar_fig = go.Figure()
    
    annual_bar_fig.add_trace(go.Bar(x=merged_dataframe['Year'], 
                        y=merged_dataframe['Profits'],
                        marker_color='green',
                        name="Profit"
                        ))
    dbConnection.close()
    
    annual_bar_fig.update_yaxes(ticksuffix="%")
    annual_bar_fig.update_layout(title='Yearly Return for ' + chart_symbol)
    #return the query ,annual bar Figure , and average return variables from function to pass up to app components
    return history_data.query(
            "Symbol == @chart_symbol"
            " and Date >= @start_date and Date <= @end_date"), annual_bar_fig, average_return

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
    #data3 = {'Cap' : ['A', 'B', 'C', ], 'non-Cap' : ['a','b','c', ]} #Headers
    #df = pd.DataFrame(data3)
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
        # [html.Tr([html.Th(col) for col in df.columns]) ] +
        # Body
        [html.Tr([
            html.Td(todf.iloc[i][col]) for col in todf.columns
        ]) for i in range(min(len(todf), max_rows))]
    ), filtered_o_data[['Name','Description']]

#symbols = dashboardData["Symbol"] + "-" + dashboardData["Security"]

#symbols.sort_values().unique()

symbols = dashboardData["Symbol"].sort_values().unique()

#regions = data["region"].sort_values().unique() 
#avocado_types = data["type"].sort_values().unique()

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
                        html.Div(children="KPI", className="menu-title"),
                        dcc.Dropdown(
                           id="kpi-filter",
                           options={
                                "Profit": 'Profit',
                                "PERatio": 'Price-to-Earnings',
                                "PEGRatio": 'PEG Ratio',
                                "PriceToBookRatio": 'Price-to-Book',
                                "Beta": 'Beta',
                                "Graham_Number": 'Graham Number'
                            },
                            value= "Profit",
                            clearable=False,
                            #searchable=False,
                            className="dropdown",
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="KPI Date Range", className="menu-title"
                        ),
                        dcc.DatePickerRange(
                            id="date-range-kpi",
                            min_date_allowed=date(1980,1,1),
                            max_date_allowed=date.today(),
                            start_date=date.today() - timedelta(days=365),
                            end_date=date.today(),
                        ),
                    ]
                ),
            ],
            className="kpi-menu",
        ),
        html.Div(
            children=[
                 html.Div(
                    children=dcc.Graph(
                        id="profit-chart",
                        config={"displayModeBar": False},
                       # figure=macd_graph(),
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),
        html.Div(children=[
                        html.P(
                            #"Some stuff to test"
                        # generate_table(df)
                        id="Company-Name",
                        className="company"
                        ),
                        html.P(id="Company-Description",
                               ), 
                        ],
                className="card-row3 wrapper"),
            
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
                        html.Div(children="Return History", className="menu-title"),
                        dcc.Dropdown(
                            id="return-filter",
                            options={
                                1: '1 Year',
                                3: '3 Year',
                                5: '5 Year',
                                10: '10 Year',
                                15: '15 Year',
                                20: '20 Year',
                                100 : 'Max'
                            },
                            value= 1,
                            clearable=False,
                            #searchable=False,
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
                        html.P(id="average-return"), 
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
                html.Div(children=[
                 html.Div(
                    children=dcc.Graph(
                        id="macd-chart",
                        config={"displayModeBar": False},
                       # figure=macd_graph(),
                    ),
                    className="card-row2 full-row",
                   ),
                 html.Div(
                    children=dcc.Graph(
                        id="annual-p-chart",
                        config={"displayModeBar": False},
                       # figure=macd_graph(),
                    ),
                    className="card-row2 full-row",
                   ),  
                ], className="cards-side2")
            ],
            className="wrapper",
        ),
    ]
)

dbConnection.close()

@app.callback(
    Output("profit-chart", "figure"),
    Input("kpi-filter", "value"),
    Input("date-range-kpi", "start_date"),
    Input("date-range-kpi", "end_date"),

)

def update_kpis(kpi,start_date, end_date): 
    
    return profit_return(kpi,start_date,end_date)


@app.callback(
    Output("price-chart", "figure"),
    Output("profit-value", "children"),
    Output("profit-value", "style"),
    Output("average-return", "children"),
    Output("volume-chart", "figure"),
    Output("macd-chart", "figure"),
    Output("annual-p-chart", "figure"),
    Input("region-filter", "value"),
    Input("return-filter", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_charts(symbol, years_return, start_date, end_date):
    filtered_data, yearly_profit_bar, average_returns = generate_query(symbol, years_return, start_date, end_date)
    begin_price = filtered_data["Adj_Close"].iloc[0]
    end_price = filtered_data["Adj_Close"].iloc[-1] 
    percent_return = round(((end_price - begin_price)/begin_price)*100,2)
    #print(percent_return)
    
    if percent_return > 0:
        dynamic_style = {'color': 'green'}
    else:
        dynamic_style = {'color': 'red'}

    profit_return_value = f"Date Range Return: {percent_return}%"
    period_of_returns = f"The Average Annual Return for {years_return} year(s) is {average_returns}%" 
    macd_chart = macd_graph(symbol, start_date, end_date)
    price_chart_figure = {
        "data": [
            {
                "x": filtered_data["Date"],
                "y": filtered_data["Adj_Close"],
                "type": "lines",
               # "hovertemplate": "$%{y:.2f}<extra></extra>",
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
    return price_chart_figure, profit_return_value, dynamic_style, period_of_returns, volume_chart_figure , macd_chart , yearly_profit_bar

@app.callback(
    Output("stock-table", "children"),
    Output("Company-Name", "children"),
    Output("Company-Description", "children"),
    Input("region-filter", "value")

)

def update_table(symbol): 
    overview_df, company_info = generate_table(symbol)
    return overview_df, company_info.iloc[0]['Name'], company_info.iloc[0]['Description']







if __name__ == "__main__":
    app.run_server(debug=True)
