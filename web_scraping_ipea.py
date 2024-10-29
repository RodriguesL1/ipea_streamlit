import requests
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np





url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"

response = requests.get(url)
if response.status_code == 200:
  soup = BeautifulSoup(response.text, 'html.parser')
  table = soup.find('table', {'id':'grd_DXMainTable'})
  df = pd.read_html(str(table), header= 0, decimal=',', thousands='.', parse_dates=True)[0]
  print('Sucesso')
else:
  print('Não encontrado, verifique a url')

df.head()


df.columns=['date','preco']

df.head()

df.info()
valor_petro = df

df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df.info()

df = df.set_index('date')

df.head()

valor_petro.preco.values.min()

def plot_series(data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Série Temporal')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()

plot_series(df, 'Preço do Petróleo ao Longo do Tempo', 'Ano', 'Preço em dólares')

df_2019 = df[df.index.year >= 2019]
df_2005 = df[df.index.year >=2005]


plot_series(df_2019, 'Preço do Petróleo ao Longo do Tempo','Ano', 'Preço em dólares')


plot_series(df_2005, 'Preço do Petróleo ao Longo do Tempo','Ano', 'Preço em dólares')




def adf_test(data):
    result = adfuller(data)
    print('Estatística ADF:', result[0])
    print('Valor-p:', result[1])
    print('Número de lags usados:', result[2])
    print('Número de observações usadas para ADF:', result[3])
    print('Valores críticos:')
    for key, value in result[4].items():
        print(f'  {key}: {value}')

data = df_2005['preco']
plot_series(data, title='Preço do Petróleo')
adf_test(data)
log_data = np.log(data)
adf_test(log_data)


resultado = seasonal_decompose(df_2005)
resultado = seasonal_decompose(df_2005['preco'], model='multiplicative', period=21)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10))

resultado.observed.plot(ax=ax1, title='Observado')
resultado.trend.plot(ax=ax2, title='Tendência')
resultado.seasonal.plot(ax=ax3, title='Sazonalidade')
resultado.resid.plot(ax=ax4, title='Resíduo')

plt.tight_layout()
plt.show()


def adf_test(data):
    result = adfuller(data)
    print('Estatística ADF:', result[0])
    print('Valor-p:', result[1])
    print('Número de lags usados:', result[2])
    print('Número de observações usadas para ADF:', result[3])
    print('Valores críticos:')
    for key, value in result[4].items():
        print(f'  {key}: {value}')



data = df_2005['preco']
plot_series(data, title='Preço do Petróleo')
adf_test(data)



ma = df_2005.rolling(100).mean()
f, ax = plt.subplots()
df_2005.plot(ax=ax, legend=False)
ma.plot(ax=ax, legend=False, color='r')
plt.tight_layout()
plt.show()

df_log = np.log(df_2005)
ma_log = df_log.rolling(50).mean()
f, ax = plt.subplots()
df_log.plot(ax=ax, legend=False)
ma_log.plot(ax=ax, legend=False, color='r')
plt.tight_layout()
plt.show()

df_s = (df_log - ma_log).dropna()
ma_s = df_s.rolling(50).mean()
std = df_s.rolling(50).std()
f, ax = plt.subplots()
df_s.plot(ax=ax, legend=False)
ma_s.plot(ax=ax, legend=False, color='r')
std.plot(ax=ax, legend=False, color='g')
plt.tight_layout()
plt.show()

adf_test(df_s)

df_diff = df_s.diff(1)
ma_diff = df_diff.rolling(50).mean()
std_diff = df_diff.rolling(50).std()

f, ax = plt.subplots()
df_diff.plot(ax=ax, legend=False)
ma_diff.plot(ax=ax, legend=False, color='r')
std_diff.plot(ax=ax, legend=False, color='g')
plt.tight_layout()
plt.show()



X_diff = df_diff['preco'].dropna().values
adf_test(X_diff)

lag_acf = acf(df_diff.dropna(), nlags=5)
lag_pacf = pacf(df_diff.dropna(), nlags=5)

#ACF
plt.plot(lag_acf)
plt.axhline(y= -1.96/(np.sqrt((len(df_diff)-1))), linestyle="--", color="grey", linewidth=0.7)
plt.axhline(y=0, linestyle="--", color="grey", linewidth=0.7)
plt.axhline(y= 1.96/(np.sqrt((len(df_diff)-1))), linestyle="--", color="grey", linewidth=0.7)

plt.title("ACF")


#PACF
plt.plot(lag_pacf)
plt.axhline(y= -1.96/(np.sqrt((len(df_diff)-1))), linestyle="--", color="grey", linewidth=0.7)
plt.axhline(y=0, linestyle="--", color="grey", linewidth=0.7)
plt.axhline(y= 1.96/(np.sqrt((len(df_diff)-1))), linestyle="--", color="grey", linewidth=0.7)

plt.title("PACF")
plt.tight_layout()
plt.show()

plot_acf(df_2005)
plot_pacf(df_2005)
plt.show()

#Vou ser bem sincero, eu assisti algumas vezes a aula sobre essa parte do ACF E PACF
# para enetender quando vale a pena fazer o ARIMA, mas não entendi. 
# Lembro que na epoca que fiz parte desse trabalho o professor Edgar 
# disse que nao valia a pena fazer o arima e acabei não fazendo

f = ["01/01/2005", "21/04/2005", "01/05/2005", "07/09/2005", "12/10/2005", "02/11/2005", "15/11/2005", "25/12/2005",
    "01/01/2006", "21/04/2006", "01/05/2006", "07/09/2006", "12/10/2006", "02/11/2006", "15/11/2006", "25/12/2006",
    "01/01/2007", "21/04/2007", "01/05/2007", "07/09/2007", "12/10/2007", "02/11/2007", "15/11/2007", "25/12/2007",
    "01/01/2008", "21/04/2008", "01/05/2008", "07/09/2008", "12/10/2008", "02/11/2008", "15/11/2008", "25/12/2008",
    "01/01/2009", "21/04/2009", "01/05/2009", "07/09/2009", "12/10/2009", "02/11/2009", "15/11/2009", "25/12/2009",
    "01/01/2010", "21/04/2010", "01/05/2010", "07/09/2010", "12/10/2010", "02/11/2010", "15/11/2010", "25/12/2010",
    "01/01/2011", "21/04/2011", "01/05/2011", "07/09/2011", "12/10/2011", "02/11/2011", "15/11/2011", "25/12/2011",
    "01/01/2012", "21/04/2012", "01/05/2012", "07/09/2012", "12/10/2012", "02/11/2012", "15/11/2012", "25/12/2012",
    "01/01/2013", "21/04/2013", "01/05/2013", "07/09/2013", "12/10/2013", "02/11/2013", "15/11/2013", "25/12/2013",
    "01/01/2014", "21/04/2014", "01/05/2014", "07/09/2014", "12/10/2014", "02/11/2014", "15/11/2014", "25/12/2014",
    "01/01/2015", "21/04/2015", "01/05/2015", "07/09/2015", "12/10/2015", "02/11/2015", "15/11/2015", "25/12/2015",
    "01/01/2016", "21/04/2016", "01/05/2016", "07/09/2016", "12/10/2016", "02/11/2016", "15/11/2016", "25/12/2016",
    "01/01/2017", "21/04/2017", "01/05/2017", "07/09/2017", "12/10/2017", "02/11/2017", "15/11/2017", "25/12/2017",
    "01/01/2018", "21/04/2018", "01/05/2018", "07/09/2018", "12/10/2018", "02/11/2018", "15/11/2018", "25/12/2018",
    "01/01/2019", "21/04/2019", "01/05/2019", "07/09/2019", "12/10/2019", "02/11/2019", "15/11/2019", "25/12/2019",
    "01/01/2020", "21/04/2020", "01/05/2020", "07/09/2020", "12/10/2020", "02/11/2020", "15/11/2020", "25/12/2020",
    "01/01/2021", "21/04/2021", "01/05/2021", "07/09/2021", "12/10/2021", "02/11/2021", "15/11/2021", "25/12/2021",
    "01/01/2022", "21/04/2022", "01/05/2022", "07/09/2022", "12/10/2022", "02/11/2022", "15/11/2022", "25/12/2022",
    "01/01/2023", "21/04/2023", "01/05/2023", "07/09/2023", "12/10/2023", "02/11/2023", "15/11/2023", "25/12/2023",
    "01/01/2024", "21/04/2024", "01/05/2024", "07/09/2024", "12/10/2024", "02/11/2024", "15/11/2024", "25/12/2024"
]
f_2019 = ["01/01/2019", "21/04/2019", "01/05/2019", "07/09/2019", "12/10/2019", "02/11/2019", "15/11/2019", "25/12/2019",
    "01/01/2020", "21/04/2020", "01/05/2020", "07/09/2020", "12/10/2020", "02/11/2020", "15/11/2020", "25/12/2020",
    "01/01/2021", "21/04/2021", "01/05/2021", "07/09/2021", "12/10/2021", "02/11/2021", "15/11/2021", "25/12/2021",
    "01/01/2022", "21/04/2022", "01/05/2022", "07/09/2022", "12/10/2022", "02/11/2022", "15/11/2022", "25/12/2022",
    "01/01/2023", "21/04/2023", "01/05/2023", "07/09/2023", "12/10/2023", "02/11/2023", "15/11/2023", "25/12/2023",
    "01/01/2024", "21/04/2024", "01/05/2024", "07/09/2024", "12/10/2024", "02/11/2024", "15/11/2024", "25/12/2024"]


feriados = pd.DataFrame({
    'holiday': 'feriado_nacional',
    'ds': pd.to_datetime(f_2019, dayfirst=True),
    'lower_window': 0,
    'upper_window': 0
})

feriados
df_prophet = df_2019.reset_index().rename(columns={'date': 'ds', 'preco': 'y'})

train_size = int(len(df_prophet) * 0.8)
train = df_prophet.iloc[:train_size]
test = df_prophet.iloc[train_size:]

m = Prophet(holidays=feriados,
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05,
    holidays_prior_scale=0.1)
m.fit(train)


future = m.make_future_dataframe(periods=len(test), freq='b')
forecast = m.predict(future)
m.plot(forecast)
plt.show()

df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='30 days')

# Avaliando a performance
df_p = performance_metrics(df_cv)
mae = mean_absolute_error(df_cv['y'], df_cv['yhat'])
mse = mean_squared_error(df_cv['y'], df_cv['yhat'])
rmse = np.sqrt(mse)
mape = np.mean(np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y'])) * 100

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")
print(f"WMAPE: {mape}%")

prev = forecast[['ds', 'yhat']]
prev.rename(columns={'ds':'date','yhat':'prev'}, inplace=True).sort_values(by='date', ascending=False, inplace= True)
prev.sort_values(by='date', ascending=False, inplace= True)

prev.info()
prev.datetime(prev.date, format='%d/%m/%Y')
prev.set_index('date', inplace=True)
prev.info()
prev.head()
df_prev = prev[prev.index.year >= 2024]

import numpy as np
from statsforecast.models import Naive, SeasonalNaive,SeasonalWindowAverage,AutoARIMA
from statsforecast import StatsForecast

valor_petro =valor_petro.rename(columns={'preco': 'y', 'date': 'ds'})

df = valor_petro
df
df["unique_id"] = "ipea"

window_size = int(0.8 * len(df))

treino= df[:window_size]
valid = df[window_size:]

h = valid['ds'].nunique()

h

def wmape(y_true, y_pred):
  return np.abs(y_true-y_pred).sum() / np.abs(y_true).sum()

model = StatsForecast(models=[Naive()], freq='D', n_jobs=-1)
model.fit(treino)

forecast_df = model.predict(h=365, level=[80])
forecast_df = forecast_df.reset_index().merge(valid, on=["ds", 'unique_id'], how='left')
forecast_df=forecast_df.dropna()
wmape1 = wmape(forecast_df['y'].values, forecast_df["Naive"].values)
print(f'WMAPE:{wmape1:.2%}')

model.plot(treino, forecast_df, level=[90], unique_ids=['ipea'], engine='matplotlib', max_insample_length=90)
plt.show()


# Ajustando o modelo e fazendo previsões
model_s = StatsForecast(models=[SeasonalNaive(season_length=7)], freq='D', n_jobs=-1)
model_s.fit(treino)

# Fazendo previsões e processando os dados
forecast_df_s = model_s.predict(h=5, level=[80])
forecast_df_s = forecast_df_s.reset_index().merge(valid, on=["ds", 'unique_id'], how='left')
forecast_df_s = forecast_df_s.dropna()

# Obtendo os valores de previsão e os valores reais
y_true = forecast_df_s['y'].values
y_pred = forecast_df_s["SeasonalNaive"].values

# Calculando as métricas
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
wmape_val = wmape(y_true, y_pred)


# Exibindo todas as métricas
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape:.2f}%")
print(f"WMAPE: {wmape_val:.2f}%")

model_s.plot(treino, forecast_df_s, level=[90], unique_ids=['ipea'], engine='matplotlib', max_insample_length=90)
plt.show()