
"""
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# Sélectionner le ticker
ticker = st.selectbox("Sélectionnez une entreprise :", ["TTE.PA", "AAPL", "GOOG"])

# Télécharger les données boursières
data = yf.download(ticker, start="2019-01-01", end="2023-12-31")

# Afficher le graphique de prix de clôture
st.line_chart(data['Close'])

# Créer un graphique avec des événements
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Prix de clôture')

# Affichage dans Streamlit
st.pyplot(plt)
"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Sélectionner le ticker
Tickers = st.selectbox("Sélectionnez une entreprise :", ["TTE.PA", "AAPL", "GOOG"])

# Téléchargement des données historiques pour TotalEnergies
Total = yf.download(Tickers, start="2010-01-01", end="2023-12-31")

# Création du DataFrame
df = Total.history(start="2000-01-01")
df = df.drop(columns=["Dividends", "Stock Splits"])


# Calcul du pourcentage de valeurs manquantes
for col in df.columns:
    percent_missing = np.mean(df[col].isnull())

# Afficher le graphique de prix de clôture
st.line_chart(df['Close'])

df['Close'].plot(figsize=(10, 6), title='Évolution du prix de clôture')
plt.ylabel('Prix de clôture (€)')
plt.xlabel('Date')
plt.grid()
plt.show()
