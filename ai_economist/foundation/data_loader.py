import pandas as pd
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_gdp_data(file_path: str) -> pd.DataFrame:
    """
    Lädt GDP-Daten aus einer CSV-Datei.

    Args:
        file_path (str): Der Pfad zur CSV-Datei.

    Returns:
        pd.DataFrame: Der DataFrame mit GDP-Daten.
    """
    try:
        df = pd.read_csv(file_path)
        # Annahme: CSV hat Spalten 'Date' und 'GDP'
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df['GDP'] = df['GDP'].fillna(method='ffill')  # Fehlende Werte auffüllen
        logger.info("GDP-Daten erfolgreich geladen und vorverarbeitet.")
        return df
    except Exception as e:
        logger.error(f"Fehler beim Laden der GDP-Daten: {e}")
        return pd.DataFrame()

def get_latest_gdp(df: pd.DataFrame, current_date: pd.Timestamp) -> float:
    """
    Gibt den neuesten verfügbaren GDP-Wert bis zum aktuellen Datum zurück.

    Args:
        df (pd.DataFrame): Der DataFrame mit GDP-Daten.
        current_date (pd.Timestamp): Das aktuelle Datum.

    Returns:
        float: Der neueste GDP-Wert.
    """
    try:
        latest_data = df[df['Date'] <= current_date]
        if not latest_data.empty:
            return latest_data.iloc[-1]['GDP']
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Fehler beim Abrufen des neuesten GDP-Werts: {e}")
        return 0.0
