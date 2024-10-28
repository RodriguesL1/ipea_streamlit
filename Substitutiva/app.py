import streamlit as st
import requests
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px


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
valor_petro = df
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df = df.set_index('date')
#st.dataframe(df.head(5))
df_2019 = df[df.index.year >= 2019]
df_2000 = df[df.index.year >= 2000]

st.set_page_config(
    page_title="Analíse de Preço do Barril de Petróleo",
    page_icon="⚫",
)
#EXPLICAÇÃO SOBRE PETROLEO
st.title('Preço do Petróleo')
st.write('Os dados são lidos do site da IPEA (http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view)', unsafe_allow_html=True)
st.header('Histórico')
st.write("O petróleo é uma commodity de alto valor agregado e que, por essa razão, é elemento central de diversas disputas geopolíticas no mundo e por conta disso muitos fatores influenciam o preço do petroleo.")


fig = px.line(df_2000, title='Evolução do Preço do Barril do Petróleo ao Longo dos Anos')
fig.update_layout(
    xaxis_title="Ano",
    yaxis_title="Preço ($)",
    width=800 
)
st.plotly_chart(fig, use_container_width=True)

st.write('Com o passar dos anos conseguimos ver essas variações nos preços e junto com elas algum acontecimento histórico no mundo.')

st.write('O aumento que enxergamos entre 2003 e 2008, foi um reflexo de uma combinações de alta demanda global, limitações de produção, especulação financeira e incertezas econômicas e geopolíticas, como a guerra do Iraque, o crescimento economico da China e da India, entre outros. ')

st.write('Porém, com a crise financeira que chegou rapidadmente no final de 2008, provocado pela quebra do banco Lehman Brothers,' 
          ' fez com que os preços despencassem, chegando a U$33,73 no mês de Dezembro do mesmo ano.')

st.write('A crise foi de curta duração e logo os preços voltaram a subir aceleradamente, em '
          'consequência do aumento significativo da demanda dos países asiáticos, com destaque para a China, e da ação '
          'da OPEP (Organização dos Países Exportadores de Petróleo) em cortar mais de quatro milhões de barris por dia no início de 2009.')

st.write('Por volta de 2011, os preços do petróleo rodeava os valores de 100 dolares, sustentado por uma demanda global. '
'Vale ressaltar que após a crise financeira, os países produtores de petróleo aumentaram significativamente a produção') 

st.write('O colapso de preços no final de 2014 é uma consequência imediata do profundo desequilíbrio entre oferta e '
'demanda global de petróleo em que um grande excesso de capacidade de produção contrasta com uma desaceleração acentuada da demanda.')


fig = px.line(df_2019, title='Evolução do Preço do Barril do Petróleo ao Longo dos Anos')
fig.update_layout(
    xaxis_title="Ano",
    yaxis_title="Preço ($)",
    width=800 
)
st.plotly_chart(fig, use_container_width=True)