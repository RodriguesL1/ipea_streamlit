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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import datetime

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
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
valor_petro = df
df = df.set_index('date')
#st.dataframe(df.head(5))
df_2018 = df[df.index.year >= 2018]
df_2019 = df[df.index.year >= 2019]
df_2000 = df[df.index.year >= 2000]

st.set_page_config(
    page_title="Analíse de Preço do Barril de Petróleo",
    page_icon="⚫",
    menu_items={
        'Get Help': 'mailto:leo2000sr@gmail.com',
        'Report a bug': "mailto:leo2000sr@gmail.com",
        'About': ''' Imagine que você foi escalado como cientista de dados em uma grande
        empresa de petróleo e precisa criar um modelo preditivo para garantir qual será
        a previsão do preço do petróleo em dólar e instanciar esse modelo preditivo em
        uma aplicação para auxiliar na tomada de decisão.
        Utilize o Streamlit para realizar a interface visual da aplicação e não se
        esqueça de realizar o deploy do modelo nessa aplicação'''}
)
#EXPLICAÇÃO SOBRE PETROLEO
st.title('Preço do Barril do Petróleo')
st.write('Os dados são lidos do site da IPEA (http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view)', unsafe_allow_html=True)
st.header('Histórico',divider='gray')
st.write("O petróleo é uma commodity de alto valor agregado e que, por essa razão, é elemento central de diversas disputas geopolíticas no mundo e por conta disso muitos fatores influenciam o preço do petróleo.")


fig = px.line(df_2000, title='Evolução do Preço do Barril do Petróleo ao Longo dos Anos')
fig.update_layout(
    xaxis_title="Ano",
    yaxis_title="Preço ($)",
    width=1000 
)
st.plotly_chart(fig, use_container_width=True)

st.write('Com o passar dos anos conseguimos ver essas variações nos preços e junto com elas algum acontecimento histórico no mundo.')

st.write('''O aumento que enxergamos entre 2003 e 2008, foi um reflexo de uma combinações de alta demanda global, limitações de produção, especulação financeira e incertezas econômicas e geopolíticas, como a guerra do Iraque, o crescimento econômico da China e da Índia, entre outros. ''')

st.write('''Porém, com a crise financeira que chegou rapidamente no final de 2008, provocado pela quebra do banco Lehman Brothers, 
          fez com que os preços despencassem, chegando a U$33,73 no mês de dezembro do mesmo ano.''')

st.write('''A crise foi de curta duração e logo os preços voltaram a subir aceleradamente, em 
          consequência do aumento significativo da demanda dos países asiáticos, com destaque para a China, e da ação 
          da OPEP (Organização dos Países Exportadores de Petróleo) em cortar mais de quatro milhões de barris por dia no início de 2009.''')

st.write('''Por volta de 2011, os preços do petróleo rodeavam os valores de 100 dólares, sustentado por uma demanda global. 
Vale ressaltar que após a crise financeira, os países produtores de petróleo aumentaram significativamente a produção''') 

st.write('''O colapso de preços no final de 2014 é uma consequência imediata do profundo desequilíbrio entre oferta e 
demanda global de petróleo em que um grande excesso de capacidade de produção contrasta com uma desaceleração acentuada da demanda.''')


fig = px.line(df_2018, title='Evolução do Preço do Barril do Petróleo ao Longo dos Anos (2018 - 2024)')
fig.update_layout(
    xaxis_title="Ano",
    yaxis_title="Preço ($)",
    width=1000 
)
st.plotly_chart(fig, use_container_width=True)

st.write('''Quando comparamos o preço do petróleo na época da pandemia,
         conseguimos ver umas das maiores quedas desde dezembro de 1998.''')
st.write('''No período de 2021 a 2022 teve uma recuperação da demanda global após a pandemia, combinado com algumas restrições,
         por conta da baixa capacidade de produção, políticas restritivas da OPEP,
         E as tenções geopolíticas, como por exemplo o conflito entre a Rússia e a Ucrânia.''')
st.write('''Esses fatores resultaram em uma alta volatilidade e em preços recordes (perdendo só para a crise de 2008), influenciando diretamente a inflação global e gerando
         impactos significativos nas economias dependentes de importação de energia''')

#st.divider()

st.header('Previsão',divider='gray')


#setando os feriados
f_2019 = ["01/01/2019", "21/04/2019", "01/05/2019", "07/09/2019", "12/10/2019", "02/11/2019", "15/11/2019", "25/12/2019",
    "01/01/2020", "21/04/2020", "01/05/2020", "07/09/2020", "12/10/2020", "02/11/2020", "15/11/2020", "25/12/2020",
    "01/01/2021", "21/04/2021", "01/05/2021", "07/09/2021", "12/10/2021", "02/11/2021", "15/11/2021", "25/12/2021",
    "01/01/2022", "21/04/2022", "01/05/2022", "07/09/2022", "12/10/2022", "02/11/2022", "15/11/2022", "25/12/2022",
    "01/01/2023", "21/04/2023", "01/05/2023", "07/09/2023", "12/10/2023", "02/11/2023", "15/11/2023", "25/12/2023",
    "01/01/2024", "21/04/2024", "01/05/2024", "07/09/2024", "12/10/2024", "02/11/2024", "15/11/2024", "25/12/2024"]




feriados = pd.DataFrame({
    'holiday': 'feriados_dias_uteis',
    'ds': pd.to_datetime(f_2019, dayfirst=True),
    'lower_window': 0,
    'upper_window': 0
})
df_prophet = df_2019.reset_index().rename(columns={'date': 'ds', 'preco': 'y'})

#separando treino e teste para depois ver quantod % esta acertando
train_size = int(len(df_prophet) * 0.8)
train = df_prophet.iloc[:train_size]
test = df_prophet.iloc[train_size:]

#treinamento do modelo Prophet colocando os feriados
m = Prophet(
    holidays=feriados,
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05,
    holidays_prior_scale=0.1
)
m.fit(train)

#fazendo previsões
future = m.make_future_dataframe(periods=len(test), freq='B')
forecast = m.predict(future)

#plotando a previsão
fig = plot_plotly(m, forecast)
st.plotly_chart(fig, use_container_width=True)


df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='30 days')

#avaliando a performance
df_p = performance_metrics(df_cv)
st.write("**Métricas de Performance:**")
st.dataframe(df_p)

mae = mean_absolute_error(df_cv['y'], df_cv['yhat'])
mse = mean_squared_error(df_cv['y'], df_cv['yhat'])
rmse = np.sqrt(mse)
mape = np.mean(np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y'])) * 100

st.write(f"**MAE:** {mae:.2f}", f" **MSE:** {mse:.2f}", f" **RMSE:** {rmse:.2f}", f" **MAPE:** {mape:.2f}%", )
st.write('''A nossa base de dados tem são contados dia-a-dia, pulando os finais de semana e feriados, 
         junto a isso tem apresenta bastante dados extrapolados por conta de ocorrências já citadas mais acima. 
         Conseguimos tratar esses pontos e chegamos nesse resultado de eficiência. 
         Resultado esse que considerei aceitável levando em considerando um commodity, que pode variar por muitos motivos.''')

prev = forecast[['ds', 'yhat']]
prev.rename(columns={'ds':'date','yhat':'prev'}, inplace=True)
prev.sort_values(by='date', ascending=False, inplace= True)

st.write("**DataFrame com as Previsões:**")
st.dataframe(prev)

#prev.datetime(prev.date, format='%d/%m/%Y')
prev.set_index('date', inplace=True)
df_prev = prev[prev.index.year >= 2023]

fig = px.line(df_prev, title='Previsão do Preço do Barril do Petróleo ao Longo dos Anos (2023 - 2025)')
fig.update_layout(
    xaxis_title="Ano",
    yaxis_title="Preço ($)",
    width=1000 
)
st.plotly_chart(fig, use_container_width=True)



st.divider()

st.markdown("**Referências:**")
st.markdown("*A crise do petróleo e os desafios do pré-sal: https://repositorio.fgv.br/items/eeab7e88-cadb-471c-a0c4-52075f5c3c0e*")
st.markdown('*Veja outros momentos em que o preço do petróleo passou de US$ 100: https://oglobo.globo.com/economia/veja-outros-momentos-em-que-preco-do-petroleo-passou-de-us-100-25408184*')
st.markdown('*Preços do petróleo se aproximam do fundo do poço de 2008: https://exame.com/economia/precos-do-petroleo-se-aproximam-do-fundo-do-poco-de-2008/*')