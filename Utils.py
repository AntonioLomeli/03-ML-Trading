import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt


def rsi_indicator(prices: pd.Series, rolling:int=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.

    Parameters:
    prices (pd.Series): A pandas Series containing the price data.
    rolling (int): The number of periods to use for the RSI calculation.

    Returns:
    pd.Series: A pandas Series containing the RSI values.
    """

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(rolling).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rolling).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def macd_indicator(prices: pd.Series, upper: int = 12, lower: int = 26, signal: int = 9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given price series.
    Parameters:
    prices (pd.Series): A pandas Series containing the price data.
    upper (int): The period for the short-term EMA.
    lower (int): The period for the long-term EMA.
    signal (int): The period for the signal line EMA.
    Returns:
    pd.Series: A pandas Series containing the MACD values.
    """

    ema_12 = prices.ewm(span=upper).mean()
    ema_26 = prices.ewm(span=lower).mean()

    macd = ema_12 - ema_26
    signal_ = macd.ewm(span=signal).mean()
    hist = macd - signal_
    return hist.fillna(0)

def metrics(y_true, y_pred, pipeline: ImbPipeline, title: str = "Confusion Matrix"):
    """
    Calculate the accuracy of predictions.

    Parameters:
    y_true (np.ndarray): The true labels.
    y_pred (np.ndarray): The predicted labels.

    Returns:
    float: The accuracy of the predictions.
    """

    accuracy_score_ = accuracy_score(y_true, y_pred)
    classification_report_ = classification_report(y_true, y_pred)

    print("Accuracy Train", accuracy_score_)
    print(f"Train Set Results: \n {classification_report_}" )
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    ax = disp.plot()
    ax.ax_.set_title(title)

    # Mostrar gráfico
    plt.show()
    
    
    return accuracy_score_, classification_report_, disp