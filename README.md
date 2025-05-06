# 📊 Previsão do Preço do Petróleo - Ipea + Streamlit

Este projeto apresenta uma aplicação interativa desenvolvida com **Streamlit** para previsão do preço do barril de petróleo (Brent) com base em dados históricos extraídos da plataforma **Ipeadata**. A modelagem preditiva é realizada utilizando o **Facebook Prophet**, uma poderosa biblioteca para séries temporais.

## 🚀 Funcionalidades

- Coleta automática de dados históricos do preço do petróleo via scraping da Ipeadata
- Limpeza e tratamento dos dados com pandas
- Previsão dos valores futuros com Prophet
- Visualização interativa com Streamlit: gráficos históricos, previsão e componentes do modelo

## 📦 Tecnologias Utilizadas

- Python
- Streamlit
- Prophet
- Pandas
- Plotly
- BeautifulSoup

## 🧠 Objetivo

O objetivo é fornecer uma ferramenta simples e intuitiva que auxilie na tomada de decisões com base em projeções do preço do petróleo, podendo ser usada por analistas de dados, investidores e interessados em macroeconomia.

## 📸 Captura de Tela
https://fiap-ipea.streamlit.app/
![image](https://github.com/user-attachments/assets/4e28b36b-981c-435c-ade5-ed67db3a5c29)


## 📂 Execução Local

```bash
git clone https://github.com/RodriguesL1/ipea_streamlit.git
cd ipea_streamlit
pip install -r requirements.txt
streamlit run app.py
