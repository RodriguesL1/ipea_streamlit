import pandas as pd

import requests
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError

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

uptaded_df =df.copy()
uptaded_df.head()

uptaded_df.info()

uptaded_df['Data'] = pd.to_datetime(uptaded_df['Data'])

uptaded_df.info()

uptaded_df.head()

uptaded_df.sort_values(by=['Data'], inplace=True)

valor_petro=uptaded_df

valor_petro

uptaded_df = uptaded_df.reset_index(drop=True)

uptaded_df

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(x='Data', y='Preço - petróleo bruto - Brent (FOB)', data = uptaded_df)

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

uptaded_df.index = uptaded_df.Data

uptaded_df.drop("Data", inplace=True, axis=1)

uptaded_df.head()

resultados = seasonal_decompose(uptaded_df, period=5)

fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize = (15,10))

resultados.observed.plot(ax=ax1)
resultados.trend.plot(ax=ax2)
resultados.seasonal.plot(ax=ax3)
resultados.resid.plot(ax=ax4)

plt.tight_layout()

df_filtrado = uptaded_df.loc[uptaded_df.index.year > 2018]

df_filtrado.head()

sns.lineplot(x='Data', y='Preço - petróleo bruto - Brent (FOB)', data = df_filtrado)

resultados1 = seasonal_decompose(df_filtrado, period=5)

fig, (ax1,ax2) = plt.subplots(2,1, figsize = (10,5))

resultados1.observed.plot(ax=ax1)
resultados1.trend.plot(ax=ax2)
#

plt.tight_layout()

from statsmodels.tsa.stattools import adfuller

X = uptaded_df['Preço - petróleo bruto - Brent (FOB)'].values

result = adfuller(X)

print("Teste ADF")
print(f"Teste Estatístico: {result[0]}")
print(f"P-Value: {result[1]}")
print("Valores críticos:")

for key, value in result[4].items():
  print(f"\t{key}: {value}")

ma = uptaded_df.rolling(100).mean()

f, ax = plt.subplots()
uptaded_df.plot(ax=ax, legend=False)
ma.plot(ax=ax, legend=False, color='r')
plt.tight_layout()

df_log = np.log(uptaded_df)
ma_log = df_log.rolling(5).mean()

f, ax = plt.subplots()
df_log.plot(ax=ax, legend=False)
ma_log.plot(ax=ax, legend=False, color='r')
plt.tight_layout()

df_s = (df_log - ma_log).dropna()

ma_s = df_s.rolling(5).mean()

std = df_s.rolling(5).std()

f, ax = plt.subplots()
df_s.plot(ax=ax, legend=False)
ma_s.plot(ax=ax, legend=False, color='r')
std.plot(ax=ax, legend=False, color='g')
plt.tight_layout()

result = adfuller(df_s['Preço - petróleo bruto - Brent (FOB)'])

print("Teste ADF")
print(f"Teste Estatístico: {result[0]}")
print(f"P-Value: {result[1]}")
print("Valores críticos:")

for key, value in result[4].items():
  print(f"\t{key}: {value}")

df_diff = df_s.diff(1)
ma_diff = df_diff.rolling(12).mean()

std_diff = df_diff.rolling(12).std()


f, ax = plt.subplots()
df_diff.plot(ax=ax, legend=False)
ma_diff.plot(ax=ax, legend=False, color='r')
std_diff.plot(ax=ax, legend=False, color='g')
plt.tight_layout()

X_diff = df_diff['Preço - petróleo bruto - Brent (FOB)'].dropna().values
result_diff = adfuller(X_diff)

print("Teste ADF")
print(f"Teste Estatístico: {result_diff[0]}")
print(f"P-Value: {result_diff[1]}")
print("Valores críticos:")

for key, value in result_diff[4].items():
  print(f"\t{key}: {value}")

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

lag_acf = acf(df_diff.dropna(), nlags=5)
lag_pacf = pacf(df_diff.dropna(), nlags=5)

#ACF
plt.plot(lag_acf)
#calculando nossas linhas de ponto critico a o numero de vezes que diferenciamos o nosso dataframe que foi 1. Lembrando que modificamos ele mais vezes, mas a diferenciação foi feita 1 vez
plt.axhline(y= -1.96/(np.sqrt((len(df_diff)-1))), linestyle="--", color="grey", linewidth=0.7)
plt.axhline(y=0, linestyle="--", color="grey", linewidth=0.7)
plt.axhline(y= 1.96/(np.sqrt((len(df_diff)-1))), linestyle="--", color="grey", linewidth=0.7)

plt.title("ACF")
plt.show()

#PACF
plt.plot(lag_pacf)
#calculando novamente as linhas de ponto critico
plt.axhline(y= -1.96/(np.sqrt((len(df_diff)-1))), linestyle="--", color="grey", linewidth=0.7)
plt.axhline(y=0, linestyle="--", color="grey", linewidth=0.7)
plt.axhline(y= 1.96/(np.sqrt((len(df_diff)-1))), linestyle="--", color="grey", linewidth=0.7)

plt.title("PACF")
plt.show()

plot_acf(uptaded_df)
plot_pacf(uptaded_df)
plt.show()

from prophet import Prophet as pr
from prophet.plot import plot_plotly, plot_components_plotly
from statsforecast.models import Naive, SeasonalNaive,SeasonalWindowAverage,AutoARIMA
from statsmodels.tsa.stattools import acf, pacf

valor_petro =valor_petro.rename(columns={'Preço - petróleo bruto - Brent (FOB)': 'y', 'Data': 'ds'})

valor_petro



m = pr()
m.fit(valor_petro)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)

plot_plotly(m, forecast)

plot_components_plotly(m, forecast)



from statsforecast.models import Naive, SeasonalNaive,SeasonalWindowAverage,AutoARIMA
from statsforecast import StatsForecast

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

forecast_df = model.predict(h=365, level=[90])
forecast_df = forecast_df.reset_index().merge(valid, on=["ds", 'unique_id'], how='left')
forecast_df=forecast_df.dropna()
wmape1 = wmape(forecast_df['y'].values, forecast_df["Naive"].values)
print(f'WMAPE:{wmape1:.2%}')

model.plot(treino, forecast_df, level=[80], unique_ids=['ipea'], engine='matplotlib', max_insample_length=90)
plt.show()

model_s = StatsForecast(models=[SeasonalNaive(season_length=7)], freq='D', n_jobs=-1)
model_s.fit(treino)

forecast_df_s = model_s.predict(h=5, level=[90])
forecast_df_s = forecast_df_s.reset_index().merge(valid, on=["ds", 'unique_id'], how='left')
forecast_df_s=forecast_df_s.dropna()
wmape2 = wmape(forecast_df_s['y'].values, forecast_df_s["SeasonalNaive"].values)
print(f'WMAPE:{wmape2:.2%}')

model_s.plot(treino, forecast_df_s, level=[9800], unique_ids=['ipea'], engine='matplotlib', max_insample_length=90)
plt.show()

forecast_df_s.head()
treino.head()
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import mean_absolute_error



# Gerar dados de exemplo
for lag in range(1, 4):
    valor_petro[f'preco_lag_{lag}'] = valor_petro['y'].shift(lag)
valor_petro = valor_petro.dropna()
X = valor_petro[['preco_lag_1', 'preco_lag_2', 'preco_lag_3']].values
y = valor_petro['y'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo Gradient Boosting Regressor
model1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, loss= 'squared_error')
model1.fit(X_train, y_train)

# Fazer previsões
y_pred = model1.predict(X_test)

# Avaliar o desempenho do modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Erro Percentual Absoluto Médio: {mape}')

residuals = y_test - y_pred
plt.plot(residuals)
plt.title('Gráfico de Resíduos')
plt.show()

# Plotar as previsões

plt.scatter(X_test[:, 0], y_test, color='black', label='Ground Truth')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predictions')
plt.title('Gradient Boosting Regressor Predictions')
plt.legend()
plt.show()

import joblib

joblib.dump(model,'model.joblib')

joblib.dump(forecast,'forecast.joblib')

