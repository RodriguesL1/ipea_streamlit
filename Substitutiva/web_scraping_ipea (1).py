#

#pip install requests
#pip install beautifulsoup4
#pip install bs4
#pip install datetime
#pip install pandas
#pip install streamlit
#pip install matplotlib
# python -m pip install prophet
#pip install seaborn

fig, (ax1, ax2, ax3, ax4) = plt.subplot(4, 1, figsize=(15, 10))

resultado.observed.plot(ax=ax1, title='ex')
resultado.trend.plot(ax=ax2, title='ex')
resultado.seasonal.plot(ax=ax3, title='ex')
resultado.resid.plot(ax=ax4, title='ex')

plt.tight_layout()
plt.show()

from bs4 import BeautifulSoup
import requests
from requests.exceptions import HTTPError
from datetime import datetime
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st



url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"

response = requests.get(url)
if response.status_code == 200:
  soup = BeautifulSoup(response.text, 'html.parser')
  table = soup.find('table', {'id':'grd_DXMainTable'})
  df = pd.read_html(str(table), header= 0)[0]
  print('Sucesso')
else:
  print('Não encontrado, verifique a url')

df.head()

#df1 = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', decimal=',', thousands='.', parse_dates=True)[2][1:]
df.columns=['date','preco']

df.head()

df.info()

df.date = pd.to_datetime(df.date, format='%d/%m/%Y')

df.info()

df.head()

df = df.set_index('date')

df=df.dropna()

df_prophet = df.reset_index().rename(columns={'date': 'ds', 'preco': 'y'})

df_prophet.head()

modelo = Prophet()
modelo.fit(df_prophet)

future = modelo.make_future_dataframe(periods=365)

forecast = modelo.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.title('Previsão do Preço do Petróleo')

# Mostrar os dados históricos
st.write("Dados Históricos")
st.line_chart(df['preco'])

# Mostrar as previsões futuras
st.write("Previsão Futura")
fig, ax = plt.subplots()
ax.plot(forecast['ds'], forecast['yhat'], label='Previsão')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
ax.legend()
st.pyplot(fig)

# Mostrar o DataFrame das previsões
st.write("Tabela de Previsões")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.title('Previsão do Preço do Petróleo')

# Mostrar os dados históricos
st.write("Dados Históricos")
st.line_chart(df['preco'])

# Entrada do usuário para definir o horizonte de previsão
st.write("Configurações de Previsão")
periodos = st.slider('Selecione o horizonte de previsão (dias)', min_value=30, max_value=365, value=180, step=30)

# Fazer previsões para o número de dias selecionado pelo usuário
future = modelo.make_future_dataframe(periods=periodos)
forecast = modelo.predict(future)

# Exibir as previsões
st.write(f"Previsão para os próximos {periodos} dias")
fig, ax = plt.subplots()
ax.plot(forecast['ds'], forecast['yhat'], label='Previsão')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
ax.legend()
st.pyplot(fig)

# Mostrar o DataFrame das previsões
st.write("Tabela de Previsões")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Mostrar os componentes de tendência e sazonalidade
st.write("Componentes da Previsão (Tendência e Sazonalidade)")
fig2 = modelo.plot_components(forecast)
st.pyplot(fig2)