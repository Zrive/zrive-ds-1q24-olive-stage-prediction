# Model Training

train_models.ipynb
- load_data + clean: selección de parcelas.
- call create_spine: [parcela, tstamp, target]
- call features -> train/test dataframe: [parcela, tstamp, feature1....N, target]
- train (hp tuning + feature_selection + cat. encoding) -> save model
- evaluacion -> loggear metrics


```python
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
```


```python
sys.path.insert(0,str(Path("../../").resolve()))

from src.load import load_clean_data
from src.load_meteo import load_clean_meteo_data

from src.features import create_spine, generate_target
from src.models import train_test_split, feature_label_split, baseline
from src.features import calculate_climatic_stats_time_window, calculate_week_number, calculates_days_in_phenological_state_current_and_previous

import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error)

from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.features import attach_meteo_var, attach_parcela_var


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
```


```python
def plot_feature_importance(clf, columns):
    fig, ax = plt.subplots()
    fi = pd.DataFrame(
        list(zip(columns, clf.feature_importances_)), columns=["features", "importance"]
    ).sort_values(by="importance", ascending=True)
    fi.plot(kind="barh", x="features", y="importance", ax=ax)
    return fi, fig, ax
```

## Load data


```python
DATA_PATH = os.path.join(Path("../").resolve(), "data")
PARCELAS_DATA_PATH = os.path.join(DATA_PATH, "clean_parcelas.parquet")
METEO_DATA_PATH = os.path.join(DATA_PATH, "clean_meteo.parquet")

if not os.path.isfile(PARCELAS_DATA_PATH) or not os.path.isfile(METEO_DATA_PATH):
    df_parcelas = load_clean_data()
    df_meteo = load_clean_meteo_data()

    df_parcelas.to_parquet(PARCELAS_DATA_PATH)
    df_meteo.to_parquet(METEO_DATA_PATH)
else:
    df_parcelas = pd.read_parquet(PARCELAS_DATA_PATH)
    df_meteo = pd.read_parquet(METEO_DATA_PATH)
    print("Datasets loaded.")
```

    Datasets loaded.


## Evaluation defitinion


```python
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='/home/agrande/zrive-ds-1q24-olive-stage-prediction/src/logs/evaluation.log', 
                    filemode='a')

def evaluate_classification(y_true, y_pred):
    metrics_dict = {}

    try:
        # Basic metrics
        metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
        metrics_dict['precision_macro'] = precision_score(y_true, y_pred, average='macro',zero_division=0)
        metrics_dict['recall_macro'] = recall_score(y_true, y_pred, average='macro',zero_division=0)
        metrics_dict['f1_macro'] = f1_score(y_true, y_pred, average='macro',zero_division=0)
        metrics_dict['mse'] = mean_squared_error(y_true, y_pred) 

        # Log and print computed metrics
        #logging.info("Computed Metrics: %s", metrics_dict)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics_dict['confusion_matrix'] = cm
        #logging.info("Confusion Matrix:\n%s", cm)

    except Exception as e:
        logging.error("Error computing metrics: %s", str(e))

    for key, value in metrics_dict.items():
        print(f"{key}: {value}\n")
    print("-----------------------------------------------------------------\n")
```

## Baseline


```python
baseline_parcelas = generate_target(df_parcelas)

baseline_train_df, baseline_test_df = train_test_split(baseline_parcelas, split_year=2021, max_year=2022)
baseline_result = baseline(baseline_train_df, baseline_test_df, 'target')[['codparcela','fecha','target','y_pred']]
```


```python
baseline_result

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>codparcela</th>
      <th>fecha</th>
      <th>target</th>
      <th>y_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>154178</th>
      <td>001-00003-01</td>
      <td>2022-06-29</td>
      <td>0.0</td>
      <td>0.175874</td>
    </tr>
    <tr>
      <th>156325</th>
      <td>001-00003-01</td>
      <td>2022-07-19</td>
      <td>0.0</td>
      <td>0.175874</td>
    </tr>
    <tr>
      <th>158014</th>
      <td>001-00003-01</td>
      <td>2022-08-02</td>
      <td>0.0</td>
      <td>0.175874</td>
    </tr>
    <tr>
      <th>159003</th>
      <td>001-00003-01</td>
      <td>2022-08-10</td>
      <td>0.0</td>
      <td>0.175874</td>
    </tr>
    <tr>
      <th>159662</th>
      <td>001-00003-01</td>
      <td>2022-08-17</td>
      <td>0.0</td>
      <td>0.175874</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>161869</th>
      <td>501-00275-02-00</td>
      <td>2022-09-07</td>
      <td>0.0</td>
      <td>0.175874</td>
    </tr>
    <tr>
      <th>162880</th>
      <td>501-00275-02-00</td>
      <td>2022-09-14</td>
      <td>0.0</td>
      <td>0.175874</td>
    </tr>
    <tr>
      <th>163835</th>
      <td>501-00275-02-00</td>
      <td>2022-09-21</td>
      <td>1.0</td>
      <td>0.175874</td>
    </tr>
    <tr>
      <th>164716</th>
      <td>501-00275-02-00</td>
      <td>2022-09-28</td>
      <td>2.0</td>
      <td>0.175874</td>
    </tr>
    <tr>
      <th>165577</th>
      <td>501-00275-02-00</td>
      <td>2022-10-05</td>
      <td>1.0</td>
      <td>0.565045</td>
    </tr>
  </tbody>
</table>
<p>22694 rows × 4 columns</p>
</div>




```python
evaluate_classification(baseline_result['target'],baseline_result['y_pred'].round())
```

    accuracy: 0.6592050762316031
    
    precision_macro: 0.2695155021443662
    
    recall_macro: 0.2555601855481858
    
    f1_macro: 0.2593766865068588
    
    mse: 0.47431039041156253
    
    confusion_matrix: [[9870 1840   44    0    0    0    0]
     [3020 3940  518    0    0    0    0]
     [ 202 1371 1150    0    0    0    0]
     [  13  544  128    0    0    0    0]
     [   1   42    0    0    0    0    0]
     [   0    9    0    0    0    0    0]
     [   0    2    0    0    0    0    0]]
    
    -----------------------------------------------------------------
    


## Feature engineering

### Generate new features


```python
# Generate meteo stats
# df_meteo_stats_aux = calculate_climatic_stats_time_window(df_meteo,'365D')

# Generate week number
df_parcelas = calculate_week_number(df_parcelas)

# Generate days in phenological state
df_parcelas_days = calculates_days_in_phenological_state_current_and_previous(df_parcelas)
```

    /home/agrande/zrive-ds-1q24-olive-stage-prediction/src/features.py:111: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df['days_spent'] = df.groupby(['codparcela','campaña'])['fecha'].diff().dt.days
    /home/agrande/zrive-ds-1q24-olive-stage-prediction/src/features.py:118: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df['days_in_current_state'] = df.groupby(['codparcela','campaña','period_id'])['days_spent'].cumsum()
    /home/agrande/zrive-ds-1q24-olive-stage-prediction/src/features.py:123: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df['days_in_previous_state'] = df.groupby(['codparcela','campaña'])['days_in_previous_state'].ffill()



```python
df_parcelas_days.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>codparcela</th>
      <th>fecha</th>
      <th>days_in_current_state</th>
      <th>days_in_previous_state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>428074</th>
      <td>001-00003-01</td>
      <td>2019-07-02</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428075</th>
      <td>001-00003-01</td>
      <td>2019-07-09</td>
      <td>7.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428076</th>
      <td>001-00003-01</td>
      <td>2019-07-17</td>
      <td>15.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428077</th>
      <td>001-00003-01</td>
      <td>2019-07-24</td>
      <td>22.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428078</th>
      <td>001-00003-01</td>
      <td>2019-07-31</td>
      <td>29.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428079</th>
      <td>001-00003-01</td>
      <td>2019-08-06</td>
      <td>35.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428080</th>
      <td>001-00003-01</td>
      <td>2019-08-13</td>
      <td>42.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428081</th>
      <td>001-00003-01</td>
      <td>2019-08-20</td>
      <td>49.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428082</th>
      <td>001-00003-01</td>
      <td>2019-08-28</td>
      <td>57.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428083</th>
      <td>001-00003-01</td>
      <td>2019-09-10</td>
      <td>70.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428084</th>
      <td>001-00003-01</td>
      <td>2019-09-17</td>
      <td>77.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428085</th>
      <td>001-00003-01</td>
      <td>2019-09-24</td>
      <td>84.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428086</th>
      <td>001-00003-01</td>
      <td>2019-10-02</td>
      <td>8.0</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>428087</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>15.0</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>428088</th>
      <td>001-00003-01</td>
      <td>2019-10-21</td>
      <td>27.0</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>428089</th>
      <td>001-00003-01</td>
      <td>2019-10-30</td>
      <td>9.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>428090</th>
      <td>001-00003-01</td>
      <td>2019-11-06</td>
      <td>16.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>428091</th>
      <td>001-00003-01</td>
      <td>2019-11-13</td>
      <td>23.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>428092</th>
      <td>001-00003-01</td>
      <td>2020-03-02</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428093</th>
      <td>001-00003-01</td>
      <td>2020-03-11</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428095</th>
      <td>001-00003-01</td>
      <td>2020-05-27</td>
      <td>77.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>428096</th>
      <td>001-00003-01</td>
      <td>2020-06-02</td>
      <td>6.0</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>428097</th>
      <td>001-00003-01</td>
      <td>2020-06-09</td>
      <td>13.0</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>428098</th>
      <td>001-00003-01</td>
      <td>2020-06-17</td>
      <td>21.0</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>428099</th>
      <td>001-00003-01</td>
      <td>2020-06-24</td>
      <td>28.0</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>428100</th>
      <td>001-00003-01</td>
      <td>2020-07-08</td>
      <td>14.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428101</th>
      <td>001-00003-01</td>
      <td>2020-07-14</td>
      <td>20.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428102</th>
      <td>001-00003-01</td>
      <td>2020-07-20</td>
      <td>26.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428103</th>
      <td>001-00003-01</td>
      <td>2020-07-27</td>
      <td>33.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428104</th>
      <td>001-00003-01</td>
      <td>2020-08-04</td>
      <td>41.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428105</th>
      <td>001-00003-01</td>
      <td>2020-08-11</td>
      <td>48.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428106</th>
      <td>001-00003-01</td>
      <td>2020-08-17</td>
      <td>54.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428107</th>
      <td>001-00003-01</td>
      <td>2020-08-23</td>
      <td>60.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428108</th>
      <td>001-00003-01</td>
      <td>2020-08-31</td>
      <td>68.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428109</th>
      <td>001-00003-01</td>
      <td>2020-09-07</td>
      <td>75.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428110</th>
      <td>001-00003-01</td>
      <td>2020-09-14</td>
      <td>82.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428111</th>
      <td>001-00003-01</td>
      <td>2020-09-21</td>
      <td>89.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428112</th>
      <td>001-00003-01</td>
      <td>2020-10-05</td>
      <td>103.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>428113</th>
      <td>001-00003-01</td>
      <td>2020-10-13</td>
      <td>8.0</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>428114</th>
      <td>001-00003-01</td>
      <td>2020-10-20</td>
      <td>15.0</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>428115</th>
      <td>001-00003-01</td>
      <td>2020-10-27</td>
      <td>7.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>428116</th>
      <td>001-00003-01</td>
      <td>2020-11-03</td>
      <td>14.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>428117</th>
      <td>001-00003-01</td>
      <td>2020-11-10</td>
      <td>21.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>428118</th>
      <td>001-00003-01</td>
      <td>2021-03-02</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428119</th>
      <td>001-00003-01</td>
      <td>2021-03-16</td>
      <td>14.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428120</th>
      <td>001-00003-01</td>
      <td>2021-03-23</td>
      <td>7.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>428121</th>
      <td>001-00003-01</td>
      <td>2021-04-06</td>
      <td>14.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>428122</th>
      <td>001-00003-01</td>
      <td>2021-04-13</td>
      <td>21.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>428123</th>
      <td>001-00003-01</td>
      <td>2021-05-05</td>
      <td>22.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>428124</th>
      <td>001-00003-01</td>
      <td>2021-05-12</td>
      <td>29.0</td>
      <td>21.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Tests


```python
DATA_PATH = os.path.join(Path("../").resolve(), "data")
PARCELAS_DATA_PATH = os.path.join(DATA_PATH, "clean_parcelas.parquet")
METEO_DATA_PATH = os.path.join(DATA_PATH, "clean_meteo.parquet")

df_meteo = pd.read_parquet(METEO_DATA_PATH)
```


```python
'''
METEO_COLUMNS = ['FAPAR','GNDVI','LST','NDVI','NDWI','SAVI','SIPI','SSM']

def calculate_climatic_stats_time_window(meteo_df: pd.DataFrame, rolling_window: str = '30D') -> pd.DataFrame:    
    # Work with necessary columns only
    columns_to_use = ['codparcela', 'fecha'] + METEO_COLUMNS
    meteo_df = meteo_df[columns_to_use]
    
    meteo_df.sort_values(by=['codparcela', 'fecha'], inplace=True)

    # Set 'fecha' as index
    meteo_df.set_index('fecha', inplace=True)

    # Calculate descriptive statistics using minimal necessary operations
    # .agg can be adjusted to compute only needed statistics to improve performance
    stats = meteo_df.groupby('codparcela').rolling(rolling_window).agg(['count','mean','std','min','median','max'])
    stats.columns = ['_'.join(col).strip() + '_' + rolling_window for col in stats.columns]

    # Reset index to make 'codparcela' and 'fecha' columns again
    stats.reset_index(inplace=True)

    return stats
'''
```


```python
df_meteo_stats_aux = calculate_climatic_stats_time_window(df_meteo,'365D')
```

    /tmp/ipykernel_128152/2646243341.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      meteo_df.sort_values(by=['codparcela', 'fecha'], inplace=True)



```python
METEO_DATA_PATH = os.path.join(DATA_PATH, "meteo_stats_365d.parquet")
df_meteo_stats_aux.to_parquet(METEO_DATA_PATH)
```

### NAN values


```python
PHENOLOGICAL_STATE_COLS = [f"estado_fenologico_{i}" for i in range(14, 0, -1)]

def analyse_nulls(df_:pd.DataFrame, columns_to_analyse:list):
    study_cols = {}
    for column in df_[columns_to_analyse]:
        if column not in PHENOLOGICAL_STATE_COLS:
            # Calcular el porcentaje de valores nulos
            null_perc = df_[column].isnull().sum() / len(df_) * 100
            # Calcular estadísticas descriptivas
            stats = df_[column].describe()

            if null_perc <= 100:
                study_cols[column] = null_perc
            
            # Imprimir la información de la columna
            print(f"Columna: {column}")
            print(f"Porcentaje de valores nulos: {null_perc:.2f}%")
            print("Estadísticas descriptivas:")
            print(stats)
            print('------------------------------------------------------------------------------------------------------------------------------')
    return study_cols
```


```python
analyse_nulls(df_parcelas,df_parcelas.select_dtypes(include=['number']).columns)
```

    Columna: campaña
    Porcentaje de valores nulos: 0.00%
    Estadísticas descriptivas:
    count    169168.000000
    mean       2019.926056
    std           1.379075
    min        2017.000000
    25%        2019.000000
    50%        2020.000000
    75%        2021.000000
    max        2023.000000
    Name: campaña, dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: poligono
    Porcentaje de valores nulos: 0.00%
    Estadísticas descriptivas:
    count     169168.0
    mean     23.028717
    std      31.121338
    min            1.0
    25%            6.0
    50%           13.0
    75%           29.0
    max          504.0
    Name: poligono, dtype: Float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: parcela
    Porcentaje de valores nulos: 0.00%
    Estadísticas descriptivas:
    count      169168.0
    mean     126.337942
    std       351.85712
    min             1.0
    25%            20.0
    50%            64.0
    75%           162.0
    max         10003.0
    Name: parcela, dtype: Float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: recinto
    Porcentaje de valores nulos: 0.01%
    Estadísticas descriptivas:
    count    169147.0
    mean     2.856758
    std      4.893189
    min           0.0
    25%           1.0
    50%           1.0
    75%           3.0
    max          56.0
    Name: recinto, dtype: Float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: subrecinto
    Porcentaje de valores nulos: 25.69%
    Estadísticas descriptivas:
    count    125717.0
    mean     1.853719
    std      7.311644
    min          -1.0
    25%           0.0
    50%           0.0
    75%           1.0
    max          98.0
    Name: subrecinto, dtype: Float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: porcentaje_floracion
    Porcentaje de valores nulos: 93.28%
    Estadísticas descriptivas:
    count    11366.000000
    mean        48.859024
    std         38.027721
    min          0.000000
    25%         10.000000
    50%         50.000000
    75%         90.000000
    max        100.000000
    Name: porcentaje_floracion, dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 102_coordenada_x_(utm)
    Porcentaje de valores nulos: 0.13%
    Estadísticas descriptivas:
    count    1.689440e+05
    mean     1.279728e+08
    std      7.270931e+09
    min      3.406400e+02
    25%      3.312780e+05
    50%      3.766390e+05
    75%      4.338690e+05
    max      4.145011e+11
    Name: 102_coordenada_x_(utm), dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 103_coordenada_y_(utm)
    Porcentaje de valores nulos: 0.13%
    Estadísticas descriptivas:
    count    1.689440e+05
    mean     1.266375e+09
    std      7.193020e+10
    min      4.121940e+05
    25%      4.119888e+06
    50%      4.139454e+06
    75%      4.179028e+06
    max      4.100596e+12
    Name: 103_coordenada_y_(utm), dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 104_altitud_(m)
    Porcentaje de valores nulos: 52.69%
    Estadísticas descriptivas:
    count    80027.000000
    mean       606.499817
    std        277.922028
    min         16.000000
    25%        420.000000
    50%        580.000000
    75%        800.000000
    max       3560.000000
    Name: 104_altitud_(m), dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 201_superf_cultivada_en_la_parcela_agrícola_(ha)
    Porcentaje de valores nulos: 1.48%
    Estadísticas descriptivas:
    count    166659.000000
    mean         19.127132
    std          41.132168
    min           0.000000
    25%           2.101190
    50%           6.140000
    75%          19.160000
    max        1048.000000
    Name: 201_superf_cultivada_en_la_parcela_agrícola_(ha), dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 202_pendiente_(%)
    Porcentaje de valores nulos: 34.43%
    Estadísticas descriptivas:
    count    110921.000000
    mean         12.146122
    std           9.335787
    min           0.000000
    25%           5.000000
    50%          10.000000
    75%          17.000000
    max          64.000000
    Name: 202_pendiente_(%), dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 302_densidad_(plantas_ha)
    Porcentaje de valores nulos: 5.35%
    Estadísticas descriptivas:
    count    160111.000000
    mean        172.088101
    std         218.934358
    min          44.000000
    25%         100.000000
    50%         125.000000
    75%         200.000000
    max        3125.000000
    Name: 302_densidad_(plantas_ha), dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 303_nº_pies_por_árbol
    Porcentaje de valores nulos: 22.92%
    Estadísticas descriptivas:
    count    130391.000000
    mean          2.052902
    std           0.896342
    min           1.000000
    25%           1.000000
    50%           2.000000
    75%           3.000000
    max           4.000000
    Name: 303_nº_pies_por_árbol, dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 305_diámetro_de_copa_(m)
    Porcentaje de valores nulos: 83.62%
    Estadísticas descriptivas:
    count    27713.000000
    mean        72.333961
    std        820.485535
    min          1.500000
    25%          4.000000
    50%          4.000000
    75%          5.000000
    max      10000.000000
    Name: 305_diámetro_de_copa_(m), dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 317_%_superficie_ocupada_variedad_secundaria
    Porcentaje de valores nulos: 96.06%
    Estadísticas descriptivas:
    count    6671.000000
    mean       26.303402
    std        15.844314
    min         0.000000
    25%        10.000000
    50%        29.000000
    75%        40.000000
    max        50.000000
    Name: 317_%_superficie_ocupada_variedad_secundaria, dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 306_altura_de_copa_(m)
    Porcentaje de valores nulos: 86.86%
    Estadísticas descriptivas:
    count    22225.000000
    mean       467.965165
    std       2857.830995
    min          1.200000
    25%          3.000000
    50%          3.000000
    75%          3.000000
    max      18035.000000
    Name: 306_altura_de_copa_(m), dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: estado_mayoritario
    Porcentaje de valores nulos: 0.00%
    Estadísticas descriptivas:
    count    169168.000000
    mean          8.186507
    std           3.034273
    min           1.000000
    25%           6.000000
    50%          10.000000
    75%          10.000000
    max          14.000000
    Name: estado_mayoritario, dtype: float64
    ------------------------------------------------------------------------------------------------------------------------------





    {'campaña': 0.0,
     'poligono': 0.0,
     'parcela': 0.0,
     'recinto': 0.01241369526151518,
     'subrecinto': 25.68511775276648,
     'porcentaje_floracion': 93.28123522179136,
     '102_coordenada_x_(utm)': 0.1324127494561619,
     '103_coordenada_y_(utm)': 0.1324127494561619,
     '104_altitud_(m)': 52.693771871748794,
     '201_superf_cultivada_en_la_parcela_agrícola_(ha)': 1.483141019578171,
     '202_pendiente_(%)': 34.431452757022605,
     '302_densidad_(plantas_ha)': 5.353849427787762,
     '303_nº_pies_por_árbol': 22.922183864560676,
     '305_diámetro_de_copa_(m)': 83.61806015322047,
     '317_%_superficie_ocupada_variedad_secundaria': 96.05658280525867,
     '306_altura_de_copa_(m)': 86.86217251489643,
     'estado_mayoritario': 0.0}




```python
analyse_nulls(df_parcelas,df_parcelas.select_dtypes(include=['category','object']).columns)
```

    Columna: codparcela
    Porcentaje de valores nulos: 0.00%
    Estadísticas descriptivas:
    count              169168
    unique               1876
    top       007-00018-01-01
    freq                  298
    Name: codparcela, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 105_comarca
    Porcentaje de valores nulos: 0.10%
    Estadísticas descriptivas:
    count     168993
    unique        56
    top       ESTEPA
    freq       17845
    Name: 105_comarca, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 203_orientación
    Porcentaje de valores nulos: 69.89%
    Estadísticas descriptivas:
    count      50938
    unique        23
    top       2 - NE
    freq        6348
    Name: 203_orientación, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 204_textura_del_suelo
    Porcentaje de valores nulos: 59.72%
    Estadísticas descriptivas:
    count                68148
    unique                  31
    top       Franco-arcilloso
    freq                 11613
    Name: 204_textura_del_suelo, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 206_secano_/_regadío
    Porcentaje de valores nulos: 1.98%
    Estadísticas descriptivas:
    count     165818
    unique         8
    top       Secano
    freq       54445
    Name: 206_secano_/_regadío, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 211_utilización_de_cubierta_vegetal
    Porcentaje de valores nulos: 18.60%
    Estadísticas descriptivas:
    count     137703
    unique         4
    top           Si
    freq       93107
    Name: 211_utilización_de_cubierta_vegetal, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 212_tipo_de_cubierta_vegetal
    Porcentaje de valores nulos: 45.07%
    Estadísticas descriptivas:
    count         92930
    unique           12
    top       Silvestre
    freq          72700
    Name: 212_tipo_de_cubierta_vegetal, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 214_cultivo_asociado/otro_aprovechamiento
    Porcentaje de valores nulos: 91.13%
    Estadísticas descriptivas:
    count     15010
    unique        5
    top          NO
    freq      12776
    Name: 214_cultivo_asociado/otro_aprovechamiento, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 301_marco_(m_x_m)
    Porcentaje de valores nulos: 3.59%
    Estadísticas descriptivas:
    count      163095
    unique        252
    top       10 X 10
    freq        31698
    Name: 301_marco_(m_x_m), dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 304_formación
    Porcentaje de valores nulos: 76.24%
    Estadísticas descriptivas:
    count     40196
    unique       26
    top        Vaso
    freq      24830
    Name: 304_formación, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 308_variedad_principal
    Porcentaje de valores nulos: 8.08%
    Estadísticas descriptivas:
    count              155500
    unique                 36
    top       Picual, Marteño
    freq                56505
    Name: 308_variedad_principal, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 311_fecha_de_plantación_variedad_principal
    Porcentaje de valores nulos: 0.05%
    Estadísticas descriptivas:
    count     169083
    unique       176
    top          nan
    freq      123626
    Name: 311_fecha_de_plantación_variedad_principal, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 208_riego:_procedencia_del_agua
    Porcentaje de valores nulos: 0.05%
    Estadísticas descriptivas:
    count     169083
    unique        35
    top          nan
    freq      129949
    Name: 208_riego:_procedencia_del_agua, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 209_riego:_calidad_del_agua
    Porcentaje de valores nulos: 0.05%
    Estadísticas descriptivas:
    count     169083
    unique        14
    top          nan
    freq      137772
    Name: 209_riego:_calidad_del_agua, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 313_variedad_secundaria
    Porcentaje de valores nulos: 83.77%
    Estadísticas descriptivas:
    count                     27450
    unique                       28
    top       Hojiblanca, Lucentino
    freq                       3471
    Name: 313_variedad_secundaria, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 107_zona_homogénea
    Porcentaje de valores nulos: 0.05%
    Estadísticas descriptivas:
    count     169083
    unique       396
    top          nan
    freq       86237
    Name: 107_zona_homogénea, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 120_zona_biológica_raif
    Porcentaje de valores nulos: 0.66%
    Estadísticas descriptivas:
    count                168054
    unique                   60
    top       GR/OL/07 IZNALLOZ
    freq                  13652
    Name: 120_zona_biológica_raif, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 401_estación_climática_asociada
    Porcentaje de valores nulos: 0.05%
    Estadísticas descriptivas:
    count     169083
    unique       131
    top          nan
    freq       67313
    Name: 401_estación_climática_asociada, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 402_sensor_climático_asociado
    Porcentaje de valores nulos: 67.96%
    Estadísticas descriptivas:
    count     54208
    unique       83
    top       SE011
    freq       6552
    Name: 402_sensor_climático_asociado, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 207_riego:_sistema_usual_de_riego
    Porcentaje de valores nulos: 0.05%
    Estadísticas descriptivas:
    count     169083
    unique        29
    top          nan
    freq      110229
    Name: 207_riego:_sistema_usual_de_riego, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 108_u_h_c_a_la_que_pertenece
    Porcentaje de valores nulos: 0.05%
    Estadísticas descriptivas:
    count     169083
    unique       862
    top          nan
    freq       50631
    Name: 108_u_h_c_a_la_que_pertenece, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 316_fecha_de_plantación_variedad_secundaria
    Porcentaje de valores nulos: 0.05%
    Estadísticas descriptivas:
    count     169083
    unique        46
    top          nan
    freq      162065
    Name: 316_fecha_de_plantación_variedad_secundaria, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 315_patrón_variedad_secundaria
    Porcentaje de valores nulos: 97.38%
    Estadísticas descriptivas:
    count                  4436
    unique                   11
    top       Lechin de Sevilla
    freq                   1188
    Name: 315_patrón_variedad_secundaria, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 310_patrón_variedad_principal
    Porcentaje de valores nulos: 79.55%
    Estadísticas descriptivas:
    count      34602
    unique        13
    top       Picual
    freq       14871
    Name: 310_patrón_variedad_principal, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 411_representa_a_la_u_h_c_(si/no)
    Porcentaje de valores nulos: 55.05%
    Estadísticas descriptivas:
    count     76049
    unique        1
    top          SI
    freq      76049
    Name: 411_representa_a_la_u_h_c_(si/no), dtype: object
    ------------------------------------------------------------------------------------------------------------------------------
    Columna: 109_sistema_para_el_cumplimiento_gestión_integrada
    Porcentaje de valores nulos: 29.76%
    Estadísticas descriptivas:
    count                        118830
    unique                            5
    top       Producción Integrada (PI)
    freq                         115029
    Name: 109_sistema_para_el_cumplimiento_gestión_integrada, dtype: object
    ------------------------------------------------------------------------------------------------------------------------------





    {'codparcela': 0.0,
     '105_comarca': 0.10344746051262649,
     '203_orientación': 69.88910432233047,
     '204_textura_del_suelo': 59.71578549134588,
     '206_secano_/_regadío': 1.9802799583845645,
     '211_utilización_de_cubierta_vegetal': 18.599853400170243,
     '212_tipo_de_cubierta_vegetal': 45.06644282606639,
     '214_cultivo_asociado/otro_aprovechamiento': 91.12716352974559,
     '301_marco_(m_x_m)': 3.589922443961033,
     '304_formación': 76.23900501276837,
     '308_variedad_principal': 8.079542230209023,
     '311_fecha_de_plantación_variedad_principal': 0.05024590939184716,
     '208_riego:_procedencia_del_agua': 0.05024590939184716,
     '209_riego:_calidad_del_agua': 0.05024590939184716,
     '313_variedad_secundaria': 83.7735269081623,
     '107_zona_homogénea': 0.05024590939184716,
     '120_zona_biológica_raif': 0.658516977206091,
     '401_estación_climática_asociada': 0.05024590939184716,
     '402_sensor_climático_asociado': 67.95611463160881,
     '207_riego:_sistema_usual_de_riego': 0.05024590939184716,
     '108_u_h_c_a_la_que_pertenece': 0.05024590939184716,
     '316_fecha_de_plantación_variedad_secundaria': 0.05024590939184716,
     '315_patrón_variedad_secundaria': 97.37775465809136,
     '310_patrón_variedad_principal': 79.54577697909771,
     '411_representa_a_la_u_h_c_(si/no)': 55.04528043128724,
     '109_sistema_para_el_cumplimiento_gestión_integrada': 29.756218670197676}



## ML pipeline


```python
### I won't use columns with more than 80% of nulls
PHENOLOGICAL_STATE_COLS = [f"estado_fenologico_{i}" for i in range(14, 0, -1)]

parcelas_numerical_cols = ['campaña','porcentaje_floracion','102_coordenada_x_(utm)',
        '103_coordenada_y_(utm)', '104_altitud_(m)','201_superf_cultivada_en_la_parcela_agrícola_(ha)',
        '202_pendiente_(%)', '302_densidad_(plantas_ha)', '303_nº_pies_por_árbol',
        '305_diámetro_de_copa_(m)','317_%_superficie_ocupada_variedad_secundaria',
        '306_altura_de_copa_(m)', 'estado_mayoritario']

parcelas_categorical_cols = ['codparcela', '105_comarca', '203_orientación', '204_textura_del_suelo',
       '206_secano_/_regadío', '211_utilización_de_cubierta_vegetal',
       '212_tipo_de_cubierta_vegetal',
       '214_cultivo_asociado/otro_aprovechamiento', '301_marco_(m_x_m)',
       '304_formación', '308_variedad_principal',
       '311_fecha_de_plantación_variedad_principal',
       '208_riego:_procedencia_del_agua', '209_riego:_calidad_del_agua',
       '313_variedad_secundaria', '107_zona_homogénea',
       '120_zona_biológica_raif', '401_estación_climática_asociada',
       '402_sensor_climático_asociado', '207_riego:_sistema_usual_de_riego',
       '108_u_h_c_a_la_que_pertenece',
       '316_fecha_de_plantación_variedad_secundaria',
       '315_patrón_variedad_secundaria', '310_patrón_variedad_principal',
       '411_representa_a_la_u_h_c_(si/no)',
       '109_sistema_para_el_cumplimiento_gestión_integrada']

meteo_numerical_cols = ['lat', 'lon', 'FAPAR', 'GNDVI', 'LST',
                         'NDVI', 'NDWI', 'SAVI', 'SIPI', 'SSM']

label_col = 'target'
```

### Correlation matrix - HACERLA CON TODAS LAS VARIABLES NUMÉRICAS DE CLIMA Y PARCELAS Y EL TARGET


```python
numerical_df = df_parcelas.select_dtypes(include=['number'])
filtered_numerical_df = numerical_df.loc[:, ~numerical_df.columns.str.startswith('estado_')]

def plot_correlation_matrix(df_):
    # Compute the correlation matrix
    corr = df_.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
plot_correlation_matrix(numerical_df)
plot_correlation_matrix(filtered_numerical_df)
```


    
![png](train_models_files/train_models_28_0.png)
    



    
![png](train_models_files/train_models_28_1.png)
    


### Logistic Regression


```python
SPLIT_YEAR = 2021
MAX_YEAR = 2022
LABEL_COL = 'target'

def preprocess_data(columns_parcela,columns_meteo):
    ### Load data and split in train test
    spine_df = create_spine(df_parcelas)

    if len(columns_parcela) > 0:
        spine_df = attach_parcela_var(spine_df, df_parcelas, columns_parcela)
    if len(columns_meteo) > 0:
        spine_df = attach_meteo_var(spine_df, df_meteo, columns_meteo, window_tolerance=2)

    # Split dataset into train and test
    train_df, test_df = train_test_split(spine_df, split_year=SPLIT_YEAR, max_year=MAX_YEAR)

    X_train, y_train = feature_label_split(train_df, LABEL_COL)
    X_test, y_test = feature_label_split(test_df, LABEL_COL)

    ### Apply transformations
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        #StandardScaler(),
    )
    
    #categorical_transformer = make_pipeline(
    #   SimpleImputer(strategy="most_frequent"),
    #    OneHotEncoder(),
    #)
    
    features_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                parcelas_numerical_cols,
            ),
            
            #(   "categorical", 
            #    categorical_transformer, 
            #    parcelas_categorical_cols
            #,           
        ],
    )

    X_train = features_transformer.fit_transform(X_train) 
    X_test = features_transformer.transform(X_test) 

    return X_train, y_train.array, X_test, y_test.array

```


```python
X_train, y_train, X_test, y_test = preprocess_data(columns_parcela=parcelas_numerical_cols,
                                                    columns_meteo = [])
```


```python
from sklearn.linear_model import LogisticRegression

lr = Pipeline(
    [
        ("standard_scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l1", C=1e-4, multi_class='multinomial', solver='saga')),
    ]
)
lr.fit(X_train, y_train)
train_preds_proba = lr.predict_proba(X_train)[:, 1]
test_preds_proba = lr.predict_proba(X_test)[:, 1]

train_preds = lr.predict(X_train)
test_preds = lr.predict(X_test)
```


```python
evaluate_classification(y_train, train_preds)
evaluate_classification(y_test, test_preds)
```

    accuracy: 0.6152360954168928
    
    precision_macro: 0.1440989231098046
    
    recall_macro: 0.16421547737743897
    
    f1_macro: 0.151172537636368
    
    mse: 0.5664065628030714
    
    confusion_matrix: [[53793  5121     0     0     0     0     0     0]
     [22969 15354     0     0     0     0     0     0]
     [ 4102  8894     0     0     0     0     0     0]
     [   70  1909     0     0     0     0     0     0]
     [    3   145     0     0     0     0     0     0]
     [    2    18     0     0     0     0     0     0]
     [    1     9     0     0     0     0     0     0]
     [    0     1     0     0     0     0     0     0]]
    
    -----------------------------------------------------------------
    
    accuracy: 0.579009848751319
    
    precision_macro: 0.1505581316401412
    
    recall_macro: 0.1772898286913916
    
    f1_macro: 0.16016581926810658
    
    mse: 0.674243756595146
    
    confusion_matrix: [[10635  1145     0     0     0     0     0]
     [ 4958  2534     0     0     0     0     0]
     [ 1012  1720     0     0     0     0     0]
     [   22   664     0     0     0     0     0]
     [    1    42     0     0     0     0     0]
     [    0     9     0     0     0     0     0]
     [    0     2     0     0     0     0     0]]
    
    -----------------------------------------------------------------
    


### Random Forest


```python
X_train, y_train, X_test, y_test = preprocess_data(columns_parcela=parcelas_numerical_cols,
                                                    columns_meteo = [])
```


```python
rf = RandomForestClassifier(500)
rf.fit(X_train, y_train)

train_preds_proba = rf.predict_proba(X_train)[:, 1]
test_preds_proba = rf.predict_proba(X_test)[:, 1]

train_preds = rf.predict(X_train)
test_preds = rf.predict(X_test)

evaluate_classification(y_train, train_preds)
evaluate_classification(y_test, test_preds)
```

    accuracy: 0.8052157201199385
    
    precision_macro: 0.8249142453615488
    
    recall_macro: 0.661458834418324
    
    f1_macro: 0.716607641282248
    
    mse: 0.28045839969392566
    
    confusion_matrix: [[54303  3980   544    74    13     0     0     0]
     [12853 24798   601    67     4     0     0     0]
     [ 1213  1729  9981    67     4     2     0     0]
     [  202   252   201  1312    11     0     1     0]
     [   27    10    10     6    95     0     0     0]
     [    8     4     3     0     0     5     0     0]
     [    2     1     3     0     0     0     4     0]
     [    0     0     0     0     0     0     0     1]]
    
    -----------------------------------------------------------------
    
    accuracy: 0.6606577558916638
    
    precision_macro: 0.2840574605028389
    
    recall_macro: 0.24602073261314544
    
    f1_macro: 0.25542471163740227
    
    mse: 0.50787020752726
    
    confusion_matrix: [[10060  1485   216    18     0     1     0     0]
     [ 2956  3708   750    77     1     0     0     0]
     [  290  1173  1213    55     1     0     0     0]
     [   57   319   263    44     1     1     1     0]
     [    5    25     9     3     0     1     0     0]
     [    0     8     0     0     0     1     0     0]
     [    0     0     0     0     0     0     0     0]
     [    0     1     1     0     0     0     0     0]]
    
    -----------------------------------------------------------------
    



```python
rf_fi, fig, ax = plot_feature_importance(rf, parcelas_numerical_cols)
```


    
![png](train_models_files/train_models_37_0.png)
    


### XGBoost


```python
X_train, y_train, X_test, y_test = preprocess_data(columns_parcela=parcelas_numerical_cols,
                                                    columns_meteo = [])


lr_best = 0.05
max_depth_best = 5
n_estimators_best = 50

gbt = GradientBoostingClassifier(
    learning_rate=lr_best, max_depth=max_depth_best, n_estimators=n_estimators_best
)
gbt.fit(X_train, y_train)
train_preds = gbt.predict(X_train)
test_preds = gbt.predict(X_test)

evaluate_classification(y_train, train_preds)
evaluate_classification(y_test, test_preds)


gbt_fi, fig, ax = plot_feature_importance(gbt, parcelas_numerical_cols)

```

    accuracy: 0.6982320648450498
    
    precision_macro: 0.6908264075803443
    
    recall_macro: 0.4592445264997421
    
    f1_macro: 0.4842432477541744
    
    mse: 0.37838439020918047
    
    confusion_matrix: [[53073  5565   253    14     0     9     0     0]
     [16113 19678  2507     6     2    17     0     0]
     [  696  6676  5580    25    11     8     0     0]
     [   50  1156   670   100     1     2     0     0]
     [    1    89    23     0    34     1     0     0]
     [    1    11     1     0     0     7     0     0]
     [    0     7     1     0     0     0     2     0]
     [    0     0     0     0     0     0     0     1]]
    
    -----------------------------------------------------------------
    
    accuracy: 0.6629880408019697
    
    precision_macro: 0.334505831542235
    
    recall_macro: 0.24777282189704683
    
    f1_macro: 0.25450530149198214
    
    mse: 0.4805223355610271
    
    confusion_matrix: [[10462  1297    20     0     0     1     0]
     [ 3467  3656   363     0     0     6     0]
     [  248  1519   955     6     1     3     0]
     [   16   552   110     6     2     0     0]
     [    1    39     1     2     0     0     0]
     [    0     9     0     0     0     0     0]
     [    0     2     0     0     0     0     0]]
    
    -----------------------------------------------------------------
    



    
![png](train_models_files/train_models_39_1.png)
    




# Functions


```python

```
