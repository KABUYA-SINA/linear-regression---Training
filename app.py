import os
from types import NotImplementedType
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# data
import numpy as np
import pandas as pd

# machine learning
import keras

# Visualisation données
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import yfinance as yf
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

style.use("ggplot")

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


# Obtention données
ticker = 'BK'
gettin_data = yf.download(ticker, start='2020-01-01', end='2024-12-01')
copy_training_df = gettin_data.copy()

# Colonnes souhaitées
open_column = gettin_data['Open']
high_column = gettin_data['High']
low_column = gettin_data['Low']
close_column = gettin_data['Close']
volume_column = gettin_data['Volume']

# Calcule PCT & PCT CHANGE
copy_training_df["PCT"] = (high_column - close_column) / close_column * 100
copy_training_df["PCT_CHANGE"] = (close_column - open_column) / open_column * 100

copy_training_df.head(200)
print('Le nombre total des colonnes est de : {0}\n\n'.format(len(copy_training_df.index)))
copy_training_df.describe(include='all')

# Corrélation
copy_training_df.corr(numeric_only=True)
feature_mean = copy_training_df.mean(numeric_only=True)
feature_std = copy_training_df.std(numeric_only=True)
numerical_features = copy_training_df.select_dtypes('number').columns

# Normalisation
normalized_dataset = (
    copy_training_df[numerical_features] - feature_mean
) / feature_std

# Choix des colonnes souhaitées
selected_data = normalized_dataset[["High", "Low", "Open", "PCT_CHANGE", "PCT", "Volume"]].copy()
selected_data.corr(numeric_only=True)

# les valeurs manquantes
selected_data.ffill(inplace=True)

sns.pairplot(selected_data[["High", "Low", "Open", "PCT_CHANGE", "PCT", "Volume"]])
missing_values = copy_training_df.isnull().sum().sum()
print("Les données relatives aux caractéristiques sont-elles manquantes ? \t\t\t\tReponse:", "Non" if missing_values == 0 else "Oui")
