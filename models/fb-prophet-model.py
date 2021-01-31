import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt

# Read file
df = pd.read_csv(r'C:\Users\jaybi\Dev\python\forecasting-app\data\peyton-manning-wiki-log-data.csv', encoding='utf8');

# Set up Prophet model
model = Prophet();
model.fit(df);

# Make predictions
future = model.make_future_dataframe(periods=365);
future.tail();
forecast = model.predict(future);
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail();

# Plot results
fig1 = model.plot(forecast);
fig2 = model.plot_components(forecast)
plt.show();