df_arima = df['Price']


train_size = int(len(df_arima) * 0.8)
train, test = df_arima[:train_size], df_arima[train_size:]


model_arima = ARIMA(train, order=(1, 1, 1))  # Os parâmetros (p,d,q) podem ser ajustados
model_arima_fit = model_arima.fit()

predictions_arima = model_arima_fit.forecast(steps=len(test))

plt.figure(figsize=(10,6))
plt.plot(test.index, test, label='Dados Reais')
plt.plot(test.index, predictions_arima, color='red', label='Previsão ARIMA')
plt.legend()
plt.title('Previsão ARIMA vs Dados Reais')
plt.show()


mae_arima = mean_absolute_error(test, predictions_arima)
rmse_arima = math.sqrt(mean_squared_error(test, predictions_arima))
print(f'MAE ARIMA: {mae_arima}')
print(f'RMSE ARIMA: {rmse_arima}')




from prophet.plot import plot_plotly, plot_components_plotly
plot_plotly(m, forecast)

fig = plot_components_plotly(m, forecast)
fig.show()


from prophet.diagnostics import cross_validation

df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')
mae_cv = mean_absolute_error(df_cv['y'], df_cv['yhat'])
mse_cv = mean_squared_error(df_cv['y'], df_cv['yhat'])
rmse_cv = np.sqrt(mse_cv)
mape_cv = np.mean(np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y'])) * 100


print(f"MAE (CV): {mae_cv}")
print(f"MSE (CV): {mse_cv}")
print(f"RMSE (CV): {rmse_cv}")
print(f"MAPE (CV): {mape_cv}%")

plt.figure(figsize=(12, 6))
plt.plot(df_cv['ds'], df_cv['y'], label='Valores Reais', color='blue')
plt.plot(df_cv['ds'], df_cv['yhat'], label='Previsões', color='orange')
plt.fill_between(df_cv['ds'], df_cv['yhat_lower'], df_cv['yhat_upper'], color='orange', alpha=0.2)
plt.xlabel('Data')
plt.ylabel('Preço')
plt.title('Comparação entre Valores Reais e Previsões')
plt.legend()
plt.show()

model = Prophet()
model.fit(df_cv)
future = model.make_future_dataframe(periods=365)
forecast_final = model.predict(future)

# Visualizando as previsões finais
model.plot(forecast_final)
plt.show()


train_size = int(len(df_cv) * 0.8)
train = df_cv.iloc[:train_size]
test = df_cv.iloc[train_size:]

model = Prophet()
model.fit(train)


future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

forecast_test = forecast.iloc[-len(test):]


mae = mean_absolute_error(test['y'], forecast_test['yhat'])
mse = mean_squared_error(test['y'], forecast_test['yhat'])
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test['y'] - forecast_test['yhat']) / test['y'])) * 100

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")







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

m = Prophet(Prophet(holidays=feriados,
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05,
    holidays_prior_scale=0.1))
m.fit(train)


future = m.make_future_dataframe(periods=len(test), freq='b')
forecast = m.predict(future)
m.plot(forecast)
plt.show()
forecast_test = forecast.iloc[-len(test):]



#mae = mean_absolute_error(test['y'], forecast_test['yhat'])
#mse = mean_squared_error(test['y'], forecast_test['yhat'])
#rmse = np.sqrt(mse)
#mape = np.mean(np.abs((test['y'] - forecast_test['yhat']) / test['y'])) * 100
#wmape = np.sum(np.abs(test['y'] - forecast_test['yhat'])) / np.sum(np.abs(test['y'])) * 100

st.subheader('Previsão')
fig1 = m.plot(forecast)
st.pyplot(fig1)
st.subheader('Componentes da Previsão')
fig2 = m.plot_components(forecast)
st.pyplot(fig2)

