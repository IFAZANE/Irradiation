import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from pmdarima.arima import auto_arima
st.set_option('deprecation.showPyplotGlobalUse', False)
# Fonction pour générer des données factices
def generate_data(start_date, end_date, num_days):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    irradiance = np.random.randint(0, 100, size=num_days)
    df = pd.DataFrame({'Date': dates[:num_days], 'Irradiation': irradiance})
    return df

# Fonction pour prédire l'irradiation solaire avec un modèle de régression polynomiale
def predict_irradiation_polynomial(df, num_days):
    X = df['Date'].astype(int).values.reshape(-1, 1)
    y = df['Irradiation'].values

    # Création de features polynomiaux
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)

    # Entraînement du modèle
    model = LinearRegression()
    model.fit(X_poly, y)

    # Génération des dates futures
    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=num_days, freq='D')
    future_X = future_dates.astype(int).values.reshape(-1, 1)
    future_X_poly = poly_features.transform(future_X)

    # Prédictions
    predictions = model.predict(future_X_poly)

    future_df = pd.DataFrame({'Date': future_dates, 'Irradiation': predictions})
    return future_df

# Fonction pour prédire l'irradiation solaire avec AutoARIMA
def predict_irradiation_autoarima(df, num_days):
    model = auto_arima(df['Irradiation'], seasonal=False, trace=True)
    predictions = model.predict(n_periods=num_days)
    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=num_days, freq='H')

    future_df = pd.DataFrame({'Date': future_dates, 'Irradiation': predictions})
    return future_df

# Fonction pour calculer le score de prédiction
def calculate_score(df, future_df):
    actual_values = df['Irradiation'].values
    predicted_values = future_df['Irradiation'].values
    mse = mean_squared_error(actual_values, predicted_values)
    return mse

# Affichage du dashboard
def main():
    st.title("Prédiction de l'irradiation solaire")

    # Sélection du modèle
    model_choice = st.radio("Choix du modèle de prédiction", ("Régression polynomiale", "AutoARIMA"))

    # Sélection des dates
    start_date = st.date_input("Date de début", datetime(2022, 1, 1))
    end_date = st.date_input("Date de fin", datetime(2022, 12, 31))

    # Calcul du nombre de jours entre les dates
    num_days = (end_date - start_date).days + 1

    # Génération des données
    df = generate_data(start_date, end_date, num_days)

    # Affichage des données
    st.subheader("Données historiques d'irradiation solaire")
    st.dataframe(df)

    # Prédiction de l'irradiation solaire
    if model_choice == "Régression polynomiale":
        future_df = predict_irradiation_polynomial(df, num_days)
    else:
        future_df = predict_irradiation_autoarima(df, num_days)

    # Calcul du score de prédiction
    score = calculate_score(df, future_df)

    # Affichage du score
    st.subheader("Score de prédiction (MSE)")
    st.write(f"Mean Squared Error: {score}")

    # Affichage des prédictions
    st.subheader("Prédictions de l'irradiation solaire")
    st.dataframe(future_df)

    # Affichage des graphiques
    st.subheader("Graphique d'irradiation solaire")
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Irradiation'], label='Données historiques', color='blue')
    plt.plot(future_df['Date'], future_df['Irradiation'], label='Prédictions', linestyle='--', color='red')
    plt.xlabel('Date')
    plt.ylabel('Irradiation solaire')
    plt.title('Prédiction de l\'irradiation solaire')
    plt.legend()
    st.pyplot()

if __name__ == "__main__":
    main()
