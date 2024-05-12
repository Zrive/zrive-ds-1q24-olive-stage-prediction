```python
from load import *
from load_meteo import *
from create_spine import *
from metrics import *
from models import *
from features import *
```

## Create a spine


```python
raw_data = load_raw_data(PARCELAS_DATA_PATH)
if raw_data is not None:
    print("Data loaded successfully.")
    print("Initial shape:", raw_data.shape)
    # Cleaning and preparing the data
    cleaned_data = clean_data(raw_data)
    print("Data cleaned.")
    print("Final shape:", cleaned_data.shape)
    display(cleaned_data.head())
else:
    print("Failed to load data, please check the path and the file.")
```

    INFO:root:Loading dataset from /Users/alvaroleal/zrive-ds-1q24-olive-stage-prediction/src/../data/muestreos_parcelas_2023.parquet
    INFO:root:Cleaning dataset
    INFO:root:Dataset shape before cleaning: (679056, 61)
    INFO:root:Filtering parcelas dataset by date
    INFO:root:Dataset shape: (190830, 61)
    INFO:root:Removing rows that have all null phenological states
    INFO:root:Dataset shape: (186521, 61)


    Data loaded successfully.
    Initial shape: (679056, 61)


    INFO:root:Creating majority phenological state column
    INFO:root:Dataset shape: (186414, 62)
    INFO:root:Removing rows with multiple provinces for the same codparcela
    INFO:root:Dataset shape: (182793, 62)
    INFO:root:remove_highly_spaced_samples_for_codparcela_in_campaign
    INFO:root:Dataset shape: (182522, 62)
    INFO:root:remove_codparcelas_not_in_meteo_data
    INFO:root:Loading dataset from /Users/alvaroleal/zrive-ds-1q24-olive-stage-prediction/src/../data/meteo.parquet
    INFO:root:Dataset shape: (165847, 62)


    Data cleaned.
    Final shape: (165847, 62)



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
      <th>generated_muestreos</th>
      <th>codparcela</th>
      <th>provincia</th>
      <th>municipio</th>
      <th>fecha</th>
      <th>campaña</th>
      <th>poligono</th>
      <th>parcela</th>
      <th>recinto</th>
      <th>subrecinto</th>
      <th>...</th>
      <th>207_riego:_sistema_usual_de_riego</th>
      <th>108_u_h_c_a_la_que_pertenece</th>
      <th>316_fecha_de_plantación_variedad_secundaria</th>
      <th>315_patrón_variedad_secundaria</th>
      <th>317_%_superficie_ocupada_variedad_secundaria</th>
      <th>306_altura_de_copa_(m)</th>
      <th>310_patrón_variedad_principal</th>
      <th>411_representa_a_la_u_h_c_(si/no)</th>
      <th>109_sistema_para_el_cumplimiento_gestión_integrada</th>
      <th>estado_mayoritario</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>428074</th>
      <td>NaT</td>
      <td>001-00003-01</td>
      <td>cadiz</td>
      <td>setenil</td>
      <td>2019-07-02</td>
      <td>2019</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Otro sistema Asesorado (OA)</td>
      <td>9</td>
    </tr>
    <tr>
      <th>428075</th>
      <td>NaT</td>
      <td>001-00003-01</td>
      <td>cadiz</td>
      <td>setenil</td>
      <td>2019-07-09</td>
      <td>2019</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Otro sistema Asesorado (OA)</td>
      <td>10</td>
    </tr>
    <tr>
      <th>428076</th>
      <td>NaT</td>
      <td>001-00003-01</td>
      <td>cadiz</td>
      <td>setenil</td>
      <td>2019-07-17</td>
      <td>2019</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Otro sistema Asesorado (OA)</td>
      <td>10</td>
    </tr>
    <tr>
      <th>428077</th>
      <td>NaT</td>
      <td>001-00003-01</td>
      <td>cadiz</td>
      <td>setenil</td>
      <td>2019-07-24</td>
      <td>2019</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Otro sistema Asesorado (OA)</td>
      <td>10</td>
    </tr>
    <tr>
      <th>428078</th>
      <td>NaT</td>
      <td>001-00003-01</td>
      <td>cadiz</td>
      <td>setenil</td>
      <td>2019-07-31</td>
      <td>2019</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Otro sistema Asesorado (OA)</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 62 columns</p>
</div>



```python
raw_meteo_data = load_clean_meteo_data()
if raw_meteo_data is not None:
    print("Data loaded successfully.")
    print("Initial shape:", raw_meteo_data.shape)
    # Cleaning and preparing the data
    cleaned_meteo_data = load_clean_meteo_data()
    print("Data cleaned.")
    print("Final shape:", cleaned_meteo_data.shape)
    display(cleaned_meteo_data.head())
else:
    print("Failed to load data, please check the path and the file.")
```

    INFO:root:Loading clean meteo dataset
    INFO:root:Loading dataset from /Users/alvaroleal/zrive-ds-1q24-olive-stage-prediction/src/../data/meteo.parquet
    INFO:root:Executing clean_meteo_data
    INFO:root:Initial dataset shape: (23469820, 6)
    INFO:root:Executing drop_nans_for_indice ['SSM']
    INFO:root:Dataset shape after operation: (23469820, 6)
    INFO:root:Executing combine_indices
    INFO:root:Dataset shape after operation: (23469820, 6)
    INFO:root:Executing drop_zeros_for_indices ['NDVI', 'NDWI', 'SAVI', 'GNDVI', 'SIPI']
    INFO:root:Dataset shape after operation: (10326873, 6)
    INFO:root:Executing normalize_indice_values FAPAR, dividing by 255.0
    INFO:root:Dataset shape after operation: (10326873, 6)
    INFO:root:Loading clean meteo dataset
    INFO:root:Loading dataset from /Users/alvaroleal/zrive-ds-1q24-olive-stage-prediction/src/../data/meteo.parquet


    Data loaded successfully.
    Initial shape: (10326873, 6)


    INFO:root:Executing clean_meteo_data
    INFO:root:Initial dataset shape: (23469820, 6)
    INFO:root:Executing drop_nans_for_indice ['SSM']
    INFO:root:Dataset shape after operation: (23469820, 6)
    INFO:root:Executing combine_indices
    INFO:root:Dataset shape after operation: (23469820, 6)
    INFO:root:Executing drop_zeros_for_indices ['NDVI', 'NDWI', 'SAVI', 'GNDVI', 'SIPI']
    INFO:root:Dataset shape after operation: (10326873, 6)
    INFO:root:Executing normalize_indice_values FAPAR, dividing by 255.0
    INFO:root:Dataset shape after operation: (10326873, 6)


    Data cleaned.
    Final shape: (10326873, 6)



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
      <th>fecha</th>
      <th>codparcela</th>
      <th>lat</th>
      <th>lon</th>
      <th>indice</th>
      <th>valor</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01</td>
      <td>023-00109-00-00</td>
      <td>37.146122</td>
      <td>-2.769372</td>
      <td>LST</td>
      <td>283.720001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-01</td>
      <td>027-00047-00-00</td>
      <td>37.140686</td>
      <td>-2.772991</td>
      <td>LST</td>
      <td>283.720001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-01</td>
      <td>016-00014-00-00</td>
      <td>37.157345</td>
      <td>-2.816261</td>
      <td>LST</td>
      <td>281.823792</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-01</td>
      <td>022-00237-00-00</td>
      <td>37.146912</td>
      <td>-2.795299</td>
      <td>LST</td>
      <td>281.823792</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01</td>
      <td>012-00154-00-00</td>
      <td>37.205017</td>
      <td>-1.919212</td>
      <td>LST</td>
      <td>286.143799</td>
    </tr>
  </tbody>
</table>
</div>



```python
raw_data = clean_data(raw_data)
```

    INFO:root:Cleaning dataset
    INFO:root:Dataset shape before cleaning: (679056, 61)
    INFO:root:Filtering parcelas dataset by date
    INFO:root:Dataset shape: (190830, 61)
    INFO:root:Removing rows that have all null phenological states
    INFO:root:Dataset shape: (186521, 61)
    INFO:root:Creating majority phenological state column
    INFO:root:Dataset shape: (186414, 62)
    INFO:root:Removing rows with multiple provinces for the same codparcela
    INFO:root:Dataset shape: (182793, 62)
    INFO:root:remove_highly_spaced_samples_for_codparcela_in_campaign
    INFO:root:Dataset shape: (182522, 62)
    INFO:root:remove_codparcelas_not_in_meteo_data
    INFO:root:Loading dataset from /Users/alvaroleal/zrive-ds-1q24-olive-stage-prediction/src/../data/meteo.parquet
    INFO:root:Dataset shape: (165847, 62)



```python
spine_df = create_spine(raw_data)
spine_df
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
      <th>campaña</th>
      <th>estado_mayoritario</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50807</th>
      <td>001-00003-01</td>
      <td>2019-07-02</td>
      <td>1.0</td>
      <td>2019</td>
      <td>9</td>
    </tr>
    <tr>
      <th>52051</th>
      <td>001-00003-01</td>
      <td>2019-07-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>10</td>
    </tr>
    <tr>
      <th>53157</th>
      <td>001-00003-01</td>
      <td>2019-07-17</td>
      <td>0.0</td>
      <td>2019</td>
      <td>10</td>
    </tr>
    <tr>
      <th>54473</th>
      <td>001-00003-01</td>
      <td>2019-07-24</td>
      <td>0.0</td>
      <td>2019</td>
      <td>10</td>
    </tr>
    <tr>
      <th>55440</th>
      <td>001-00003-01</td>
      <td>2019-07-31</td>
      <td>0.0</td>
      <td>2019</td>
      <td>10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>126167</th>
      <td>504-10003-03-1</td>
      <td>2021-08-10</td>
      <td>0.0</td>
      <td>2021</td>
      <td>10</td>
    </tr>
    <tr>
      <th>128028</th>
      <td>504-10003-03-1</td>
      <td>2021-08-25</td>
      <td>0.0</td>
      <td>2021</td>
      <td>10</td>
    </tr>
    <tr>
      <th>129452</th>
      <td>504-10003-03-1</td>
      <td>2021-09-07</td>
      <td>0.0</td>
      <td>2021</td>
      <td>10</td>
    </tr>
    <tr>
      <th>131498</th>
      <td>504-10003-03-1</td>
      <td>2021-09-21</td>
      <td>1.0</td>
      <td>2021</td>
      <td>10</td>
    </tr>
    <tr>
      <th>133110</th>
      <td>504-10003-03-1</td>
      <td>2021-10-05</td>
      <td>0.0</td>
      <td>2021</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
<p>131511 rows × 5 columns</p>
</div>




```python
unique_target_values = spine_df['target'].unique()
print(unique_target_values)
```

    [1. 0. 2. 3.]


## Add features


```python
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    features_df = (
        data
        .pipe(calculate_week_number)  # Apply calculate_week_number first
        .pipe(calculates_days_in_phenological_state_current_and_previous)  # Then apply calculates_days_in_phenological_state_current_and_previous
    )
    return features_df
```


```python
features_df = add_features(spine_df)
features_df
```

    /Users/alvaroleal/zrive-ds-1q24-olive-stage-prediction/src/features.py:120: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df['days_spent'] = df.groupby(['codparcela', 'campaña'])['fecha'].diff().dt.days
    /Users/alvaroleal/zrive-ds-1q24-olive-stage-prediction/src/features.py:127: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df['days_in_current_state'] = df.groupby(['codparcela', 'campaña', 'period_id'])[
    /Users/alvaroleal/zrive-ds-1q24-olive-stage-prediction/src/features.py:140: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df['days_in_previous_state'] = df.groupby(['codparcela', 'campaña'])[





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
      <th>campaña</th>
      <th>estado_mayoritario</th>
      <th>week_number</th>
      <th>state_change</th>
      <th>days_in_current_state</th>
      <th>days_in_previous_state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65142</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>66316</th>
      <td>001-00003-01</td>
      <td>2019-10-21</td>
      <td>1.0</td>
      <td>2019</td>
      <td>11</td>
      <td>43</td>
      <td>False</td>
      <td>27.0</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>67710</th>
      <td>001-00003-01</td>
      <td>2019-10-30</td>
      <td>0.0</td>
      <td>2019</td>
      <td>12</td>
      <td>44</td>
      <td>True</td>
      <td>9.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>86993</th>
      <td>001-00003-01</td>
      <td>2020-07-08</td>
      <td>0.0</td>
      <td>2020</td>
      <td>10</td>
      <td>28</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>87432</th>
      <td>001-00003-01</td>
      <td>2020-07-14</td>
      <td>0.0</td>
      <td>2020</td>
      <td>10</td>
      <td>29</td>
      <td>False</td>
      <td>20.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>126167</th>
      <td>504-10003-03-1</td>
      <td>2021-08-10</td>
      <td>0.0</td>
      <td>2021</td>
      <td>10</td>
      <td>32</td>
      <td>True</td>
      <td>49.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>128028</th>
      <td>504-10003-03-1</td>
      <td>2021-08-25</td>
      <td>0.0</td>
      <td>2021</td>
      <td>10</td>
      <td>34</td>
      <td>False</td>
      <td>64.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>129452</th>
      <td>504-10003-03-1</td>
      <td>2021-09-07</td>
      <td>0.0</td>
      <td>2021</td>
      <td>10</td>
      <td>36</td>
      <td>False</td>
      <td>77.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>131498</th>
      <td>504-10003-03-1</td>
      <td>2021-09-21</td>
      <td>1.0</td>
      <td>2021</td>
      <td>10</td>
      <td>38</td>
      <td>False</td>
      <td>91.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>133110</th>
      <td>504-10003-03-1</td>
      <td>2021-10-05</td>
      <td>0.0</td>
      <td>2021</td>
      <td>11</td>
      <td>40</td>
      <td>True</td>
      <td>14.0</td>
      <td>91.0</td>
    </tr>
  </tbody>
</table>
<p>113938 rows × 9 columns</p>
</div>



## Add meteo features


```python
features_df = merge_and_clean(features_df, raw_meteo_data)
features_df
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
      <th>campaña</th>
      <th>estado_mayoritario</th>
      <th>week_number</th>
      <th>state_change</th>
      <th>days_in_current_state</th>
      <th>days_in_previous_state</th>
      <th>indice</th>
      <th>valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>NDVI</td>
      <td>0.355463</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>NDWI</td>
      <td>-0.484848</td>
    </tr>
    <tr>
      <th>2</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>GNDVI</td>
      <td>0.141414</td>
    </tr>
    <tr>
      <th>3</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>SAVI</td>
      <td>0.195388</td>
    </tr>
    <tr>
      <th>4</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>SIPI</td>
      <td>1.464980</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>197132</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>NDWI</td>
      <td>-0.396616</td>
    </tr>
    <tr>
      <th>197133</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>GNDVI</td>
      <td>0.058293</td>
    </tr>
    <tr>
      <th>197134</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>SAVI</td>
      <td>0.255370</td>
    </tr>
    <tr>
      <th>197135</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>SIPI</td>
      <td>1.304906</td>
    </tr>
    <tr>
      <th>197139</th>
      <td>504-10003-03-1</td>
      <td>2021-09-21</td>
      <td>1.0</td>
      <td>2021</td>
      <td>10</td>
      <td>38</td>
      <td>False</td>
      <td>91.0</td>
      <td>14.0</td>
      <td>LST</td>
      <td>296.667389</td>
    </tr>
  </tbody>
</table>
<p>119634 rows × 11 columns</p>
</div>




```python
features_df = calculate_and_merge_climatic_stats(features_df, '30D')

features_df.columns
```




    Index(['codparcela', 'fecha', 'target', 'campaña', 'estado_mayoritario',
           'week_number', 'state_change', 'days_in_current_state',
           'days_in_previous_state', 'indice', 'valor', 'valor_FAPAR_count',
           'valor_FAPAR_mean', 'valor_FAPAR_std', 'valor_FAPAR_min',
           'valor_FAPAR_median', 'valor_FAPAR_max', 'valor_GNDVI_count',
           'valor_GNDVI_mean', 'valor_GNDVI_std', 'valor_GNDVI_min',
           'valor_GNDVI_median', 'valor_GNDVI_max', 'valor_LST_count',
           'valor_LST_mean', 'valor_LST_std', 'valor_LST_min', 'valor_LST_median',
           'valor_LST_max', 'valor_NDVI_count', 'valor_NDVI_mean',
           'valor_NDVI_std', 'valor_NDVI_min', 'valor_NDVI_median',
           'valor_NDVI_max', 'valor_NDWI_count', 'valor_NDWI_mean',
           'valor_NDWI_std', 'valor_NDWI_min', 'valor_NDWI_median',
           'valor_NDWI_max', 'valor_SAVI_count', 'valor_SAVI_mean',
           'valor_SAVI_std', 'valor_SAVI_min', 'valor_SAVI_median',
           'valor_SAVI_max', 'valor_SIPI_count', 'valor_SIPI_mean',
           'valor_SIPI_std', 'valor_SIPI_min', 'valor_SIPI_median',
           'valor_SIPI_max'],
          dtype='object')




```python
features_df
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
      <th>campaña</th>
      <th>estado_mayoritario</th>
      <th>week_number</th>
      <th>state_change</th>
      <th>days_in_current_state</th>
      <th>days_in_previous_state</th>
      <th>indice</th>
      <th>...</th>
      <th>valor_SAVI_std</th>
      <th>valor_SAVI_min</th>
      <th>valor_SAVI_median</th>
      <th>valor_SAVI_max</th>
      <th>valor_SIPI_count</th>
      <th>valor_SIPI_mean</th>
      <th>valor_SIPI_std</th>
      <th>valor_SIPI_min</th>
      <th>valor_SIPI_median</th>
      <th>valor_SIPI_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>NDVI</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>1.0</td>
      <td>1.464980</td>
      <td>0.0</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>1.464980</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>NDWI</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>1.0</td>
      <td>1.464980</td>
      <td>0.0</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>1.464980</td>
    </tr>
    <tr>
      <th>2</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>GNDVI</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>1.0</td>
      <td>1.464980</td>
      <td>0.0</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>1.464980</td>
    </tr>
    <tr>
      <th>3</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>SAVI</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>1.0</td>
      <td>1.464980</td>
      <td>0.0</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>1.464980</td>
    </tr>
    <tr>
      <th>4</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>SIPI</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>1.0</td>
      <td>1.464980</td>
      <td>0.0</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>1.464980</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>119629</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>NDWI</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>1.0</td>
      <td>1.304906</td>
      <td>0.0</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>1.304906</td>
    </tr>
    <tr>
      <th>119630</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>GNDVI</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>1.0</td>
      <td>1.304906</td>
      <td>0.0</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>1.304906</td>
    </tr>
    <tr>
      <th>119631</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>SAVI</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>1.0</td>
      <td>1.304906</td>
      <td>0.0</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>1.304906</td>
    </tr>
    <tr>
      <th>119632</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>SIPI</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>1.0</td>
      <td>1.304906</td>
      <td>0.0</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>1.304906</td>
    </tr>
    <tr>
      <th>119633</th>
      <td>504-10003-03-1</td>
      <td>2021-09-21</td>
      <td>1.0</td>
      <td>2021</td>
      <td>10</td>
      <td>38</td>
      <td>False</td>
      <td>91.0</td>
      <td>14.0</td>
      <td>LST</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>119634 rows × 53 columns</p>
</div>



## Models


```python
info_cols = ['codparcela', 'fecha','campaña','indice','valor','year']
label_col = 'target'
features_cols = [col for col in features_df.columns if col not in info_cols + [label_col]]

categorical_cols = []
binary_cols = ['state_change']
meteo_cols = ['valor_FAPAR_count',
       'valor_FAPAR_mean', 'valor_FAPAR_std', 'valor_FAPAR_min',
       'valor_FAPAR_median', 'valor_FAPAR_max', 'valor_GNDVI_count',
       'valor_GNDVI_mean', 'valor_GNDVI_std', 'valor_GNDVI_min',
       'valor_GNDVI_median', 'valor_GNDVI_max', 'valor_LST_count',
       'valor_LST_mean', 'valor_LST_std', 'valor_LST_min', 'valor_LST_median',
       'valor_LST_max', 'valor_NDVI_count', 'valor_NDVI_mean',
       'valor_NDVI_std', 'valor_NDVI_min', 'valor_NDVI_median',
       'valor_NDVI_max', 'valor_NDWI_count', 'valor_NDWI_mean',
       'valor_NDWI_std', 'valor_NDWI_min', 'valor_NDWI_median',
       'valor_NDWI_max', 'valor_SAVI_count', 'valor_SAVI_mean',
       'valor_SAVI_std', 'valor_SAVI_min', 'valor_SAVI_median',
       'valor_SAVI_max', 'valor_SIPI_count', 'valor_SIPI_mean',
       'valor_SIPI_std', 'valor_SIPI_min', 'valor_SIPI_median',
       'valor_SIPI_max']
numerical_cols = [col for col in features_cols if col not in meteo_cols+ binary_cols]
```

### Baseline


```python
train_cols = numerical_cols + binary_cols + meteo_cols

train_set, test_set = train_test_split(features_df, split_year=2021, max_year=2022)
train_set
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
      <th>campaña</th>
      <th>estado_mayoritario</th>
      <th>week_number</th>
      <th>state_change</th>
      <th>days_in_current_state</th>
      <th>days_in_previous_state</th>
      <th>indice</th>
      <th>...</th>
      <th>valor_SAVI_min</th>
      <th>valor_SAVI_median</th>
      <th>valor_SAVI_max</th>
      <th>valor_SIPI_count</th>
      <th>valor_SIPI_mean</th>
      <th>valor_SIPI_std</th>
      <th>valor_SIPI_min</th>
      <th>valor_SIPI_median</th>
      <th>valor_SIPI_max</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>NDVI</td>
      <td>...</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>1.0</td>
      <td>1.464980</td>
      <td>0.0</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>NDWI</td>
      <td>...</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>1.0</td>
      <td>1.464980</td>
      <td>0.0</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>GNDVI</td>
      <td>...</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>1.0</td>
      <td>1.464980</td>
      <td>0.0</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>SAVI</td>
      <td>...</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>1.0</td>
      <td>1.464980</td>
      <td>0.0</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>001-00003-01</td>
      <td>2019-10-09</td>
      <td>0.0</td>
      <td>2019</td>
      <td>11</td>
      <td>41</td>
      <td>True</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>SIPI</td>
      <td>...</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>0.195388</td>
      <td>1.0</td>
      <td>1.464980</td>
      <td>0.0</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>1.464980</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>119629</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>NDWI</td>
      <td>...</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>1.0</td>
      <td>1.304906</td>
      <td>0.0</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>119630</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>GNDVI</td>
      <td>...</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>1.0</td>
      <td>1.304906</td>
      <td>0.0</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>119631</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>SAVI</td>
      <td>...</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>1.0</td>
      <td>1.304906</td>
      <td>0.0</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>119632</th>
      <td>504-10003-03-1</td>
      <td>2021-06-22</td>
      <td>0.0</td>
      <td>2021</td>
      <td>9</td>
      <td>25</td>
      <td>True</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>SIPI</td>
      <td>...</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>0.255370</td>
      <td>1.0</td>
      <td>1.304906</td>
      <td>0.0</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>1.304906</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>119633</th>
      <td>504-10003-03-1</td>
      <td>2021-09-21</td>
      <td>1.0</td>
      <td>2021</td>
      <td>10</td>
      <td>38</td>
      <td>False</td>
      <td>91.0</td>
      <td>14.0</td>
      <td>LST</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2021</td>
    </tr>
  </tbody>
</table>
<p>101120 rows × 54 columns</p>
</div>




```python
baseline(train_set, test_set, 'target')
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
      <th>campaña</th>
      <th>estado_mayoritario</th>
      <th>week_number</th>
      <th>state_change</th>
      <th>days_in_current_state</th>
      <th>days_in_previous_state</th>
      <th>indice</th>
      <th>...</th>
      <th>valor_SAVI_median</th>
      <th>valor_SAVI_max</th>
      <th>valor_SIPI_count</th>
      <th>valor_SIPI_mean</th>
      <th>valor_SIPI_std</th>
      <th>valor_SIPI_min</th>
      <th>valor_SIPI_median</th>
      <th>valor_SIPI_max</th>
      <th>year</th>
      <th>y_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132</th>
      <td>001-00003-02-01</td>
      <td>2022-05-10</td>
      <td>0.0</td>
      <td>2022</td>
      <td>8</td>
      <td>19</td>
      <td>True</td>
      <td>14.0</td>
      <td>21.0</td>
      <td>FAPAR</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2022</td>
      <td>0.981779</td>
    </tr>
    <tr>
      <th>133</th>
      <td>001-00003-02-01</td>
      <td>2022-06-01</td>
      <td>0.0</td>
      <td>2022</td>
      <td>9</td>
      <td>22</td>
      <td>True</td>
      <td>14.0</td>
      <td>22.0</td>
      <td>LST</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2022</td>
      <td>0.352810</td>
    </tr>
    <tr>
      <th>134</th>
      <td>001-00003-02-01</td>
      <td>2022-07-12</td>
      <td>0.0</td>
      <td>2022</td>
      <td>10</td>
      <td>28</td>
      <td>False</td>
      <td>20.0</td>
      <td>35.0</td>
      <td>NDVI</td>
      <td>...</td>
      <td>0.179505</td>
      <td>0.179505</td>
      <td>1.0</td>
      <td>1.737748</td>
      <td>0.0</td>
      <td>1.737748</td>
      <td>1.737748</td>
      <td>1.737748</td>
      <td>2022</td>
      <td>0.170684</td>
    </tr>
    <tr>
      <th>135</th>
      <td>001-00003-02-01</td>
      <td>2022-07-12</td>
      <td>0.0</td>
      <td>2022</td>
      <td>10</td>
      <td>28</td>
      <td>False</td>
      <td>20.0</td>
      <td>35.0</td>
      <td>NDWI</td>
      <td>...</td>
      <td>0.179505</td>
      <td>0.179505</td>
      <td>1.0</td>
      <td>1.737748</td>
      <td>0.0</td>
      <td>1.737748</td>
      <td>1.737748</td>
      <td>1.737748</td>
      <td>2022</td>
      <td>0.170684</td>
    </tr>
    <tr>
      <th>136</th>
      <td>001-00003-02-01</td>
      <td>2022-07-12</td>
      <td>0.0</td>
      <td>2022</td>
      <td>10</td>
      <td>28</td>
      <td>False</td>
      <td>20.0</td>
      <td>35.0</td>
      <td>GNDVI</td>
      <td>...</td>
      <td>0.179505</td>
      <td>0.179505</td>
      <td>1.0</td>
      <td>1.737748</td>
      <td>0.0</td>
      <td>1.737748</td>
      <td>1.737748</td>
      <td>1.737748</td>
      <td>2022</td>
      <td>0.170684</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>119521</th>
      <td>501-00275-02-00</td>
      <td>2022-09-28</td>
      <td>2.0</td>
      <td>2022</td>
      <td>10</td>
      <td>39</td>
      <td>False</td>
      <td>77.0</td>
      <td>28.0</td>
      <td>NDVI</td>
      <td>...</td>
      <td>0.206858</td>
      <td>0.206858</td>
      <td>1.0</td>
      <td>1.352459</td>
      <td>0.0</td>
      <td>1.352459</td>
      <td>1.352459</td>
      <td>1.352459</td>
      <td>2022</td>
      <td>0.170684</td>
    </tr>
    <tr>
      <th>119522</th>
      <td>501-00275-02-00</td>
      <td>2022-09-28</td>
      <td>2.0</td>
      <td>2022</td>
      <td>10</td>
      <td>39</td>
      <td>False</td>
      <td>77.0</td>
      <td>28.0</td>
      <td>NDWI</td>
      <td>...</td>
      <td>0.206858</td>
      <td>0.206858</td>
      <td>1.0</td>
      <td>1.352459</td>
      <td>0.0</td>
      <td>1.352459</td>
      <td>1.352459</td>
      <td>1.352459</td>
      <td>2022</td>
      <td>0.170684</td>
    </tr>
    <tr>
      <th>119523</th>
      <td>501-00275-02-00</td>
      <td>2022-09-28</td>
      <td>2.0</td>
      <td>2022</td>
      <td>10</td>
      <td>39</td>
      <td>False</td>
      <td>77.0</td>
      <td>28.0</td>
      <td>GNDVI</td>
      <td>...</td>
      <td>0.206858</td>
      <td>0.206858</td>
      <td>1.0</td>
      <td>1.352459</td>
      <td>0.0</td>
      <td>1.352459</td>
      <td>1.352459</td>
      <td>1.352459</td>
      <td>2022</td>
      <td>0.170684</td>
    </tr>
    <tr>
      <th>119524</th>
      <td>501-00275-02-00</td>
      <td>2022-09-28</td>
      <td>2.0</td>
      <td>2022</td>
      <td>10</td>
      <td>39</td>
      <td>False</td>
      <td>77.0</td>
      <td>28.0</td>
      <td>SAVI</td>
      <td>...</td>
      <td>0.206858</td>
      <td>0.206858</td>
      <td>1.0</td>
      <td>1.352459</td>
      <td>0.0</td>
      <td>1.352459</td>
      <td>1.352459</td>
      <td>1.352459</td>
      <td>2022</td>
      <td>0.170684</td>
    </tr>
    <tr>
      <th>119525</th>
      <td>501-00275-02-00</td>
      <td>2022-09-28</td>
      <td>2.0</td>
      <td>2022</td>
      <td>10</td>
      <td>39</td>
      <td>False</td>
      <td>77.0</td>
      <td>28.0</td>
      <td>SIPI</td>
      <td>...</td>
      <td>0.206858</td>
      <td>0.206858</td>
      <td>1.0</td>
      <td>1.352459</td>
      <td>0.0</td>
      <td>1.352459</td>
      <td>1.352459</td>
      <td>1.352459</td>
      <td>2022</td>
      <td>0.170684</td>
    </tr>
  </tbody>
</table>
<p>18514 rows × 55 columns</p>
</div>




```python
metrics(test_set['target'], test_set['y_pred'])
```




    {'accuracy': 0.6988765258723129,
     'mse': 0.3150995826502369,
     'mae': 0.4173216209336987}



### Logistic Regression


```python
features_and_coefs, test_predictions = logistic_regression_model(train_set, test_set, 'target', train_cols, penalty='l1', C=0.1, solver='liblinear')

print(features_and_coefs)
```

                       Feature  Coefficient
    0       estado_mayoritario     2.668041
    1              week_number    -1.210328
    2    days_in_current_state     0.043768
    3   days_in_previous_state     0.041216
    4             state_change    -0.028551
    5        valor_FAPAR_count     0.191106
    6         valor_FAPAR_mean    -0.028148
    7          valor_FAPAR_std    -0.056002
    8          valor_FAPAR_min    -0.011335
    9       valor_FAPAR_median     0.000000
    10         valor_FAPAR_max    -0.143444
    11       valor_GNDVI_count     0.000000
    12        valor_GNDVI_mean     0.000000
    13         valor_GNDVI_std     0.000000
    14         valor_GNDVI_min     0.004142
    15      valor_GNDVI_median    -0.042084
    16         valor_GNDVI_max     0.004080
    17         valor_LST_count    -0.185848
    18          valor_LST_mean     0.000000
    19           valor_LST_std    -0.003068
    20           valor_LST_min     0.067251
    21        valor_LST_median     0.000000
    22           valor_LST_max     0.000000
    23        valor_NDVI_count     0.000000
    24         valor_NDVI_mean     0.000000
    25          valor_NDVI_std    -0.150333
    26          valor_NDVI_min     0.000000
    27       valor_NDVI_median    -0.013402
    28          valor_NDVI_max     0.000000
    29        valor_NDWI_count     0.000000
    30         valor_NDWI_mean     0.000000
    31          valor_NDWI_std    -0.463117
    32          valor_NDWI_min    -0.510944
    33       valor_NDWI_median     0.662096
    34          valor_NDWI_max     0.000000
    35        valor_SAVI_count     0.000000
    36         valor_SAVI_mean     0.000000
    37          valor_SAVI_std     0.000000
    38          valor_SAVI_min     0.009311
    39       valor_SAVI_median    -0.017087
    40          valor_SAVI_max     0.014898
    41        valor_SIPI_count    -0.038300
    42         valor_SIPI_mean     0.000000
    43          valor_SIPI_std     0.007822
    44          valor_SIPI_min     0.000000
    45       valor_SIPI_median    -0.053542
    46          valor_SIPI_max     1.045126



```python
metrics(test_set['target'], test_set['y_pred'])
```




    {'accuracy': 0.6069460948471427,
     'mse': 0.6378416333585395,
     'mae': 0.4740736739764502}



### Gradient Boosting Classifier


```python
features_and_importances, test_predictions = gradient_boosting_model(train_set, test_set, 'target', features_cols, n_estimators=100, learning_rate=0.1, max_depth=3)
print(features_and_importances)
```

                       Feature  Importance
    0       estado_mayoritario    0.603142
    1              week_number    0.208305
    3    days_in_current_state    0.093132
    4   days_in_previous_state    0.012550
    34          valor_NDWI_max    0.010473
    2             state_change    0.008975
    32          valor_NDWI_min    0.008054
    26          valor_NDVI_min    0.008034
    27       valor_NDVI_median    0.005484
    45       valor_SIPI_median    0.005200
    28          valor_NDVI_max    0.004949
    46          valor_SIPI_max    0.003695
    44          valor_SIPI_min    0.003366
    43          valor_SIPI_std    0.003226
    42         valor_SIPI_mean    0.002964
    33       valor_NDWI_median    0.002887
    30         valor_NDWI_mean    0.002475
    20           valor_LST_min    0.002439
    24         valor_NDVI_mean    0.001999
    22           valor_LST_max    0.001596
    18          valor_LST_mean    0.001032
    21        valor_LST_median    0.000829
    8          valor_FAPAR_min    0.000792
    25          valor_NDVI_std    0.000640
    31          valor_NDWI_std    0.000599
    5        valor_FAPAR_count    0.000522
    9       valor_FAPAR_median    0.000418
    19           valor_LST_std    0.000406
    17         valor_LST_count    0.000389
    11       valor_GNDVI_count    0.000338
    10         valor_FAPAR_max    0.000269
    7          valor_FAPAR_std    0.000249
    6         valor_FAPAR_mean    0.000218
    35        valor_SAVI_count    0.000128
    37          valor_SAVI_std    0.000076
    13         valor_GNDVI_std    0.000074
    14         valor_GNDVI_min    0.000020
    12        valor_GNDVI_mean    0.000016
    15      valor_GNDVI_median    0.000013
    16         valor_GNDVI_max    0.000010
    23        valor_NDVI_count    0.000008
    39       valor_SAVI_median    0.000006
    38          valor_SAVI_min    0.000005
    41        valor_SIPI_count    0.000000
    40          valor_SAVI_max    0.000000
    36         valor_SAVI_mean    0.000000
    29        valor_NDWI_count    0.000000



```python
metrics(test_set['target'], test_set['y_pred'])
```




    {'accuracy': 0.7674192502970725,
     'mse': 0.2790860970076699,
     'mae': 0.24808253213784162}



### Gradient Boosting Regressor


```python
features_and_importances, test_predictions = gradient_boosting_model_gbm(train_set, test_set, 'target', train_cols, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8)
print(features_and_importances)
```

                       Feature  Importance
    0       estado_mayoritario    0.773902
    1              week_number    0.115073
    2    days_in_current_state    0.053046
    4             state_change    0.008694
    26          valor_NDVI_min    0.007805
    3   days_in_previous_state    0.006178
    34          valor_NDWI_max    0.005803
    27       valor_NDVI_median    0.004293
    46          valor_SIPI_max    0.003579
    44          valor_SIPI_min    0.003055
    45       valor_SIPI_median    0.002842
    28          valor_NDVI_max    0.002540
    33       valor_NDWI_median    0.002360
    32          valor_NDWI_min    0.001998
    30         valor_NDWI_mean    0.001892
    24         valor_NDVI_mean    0.001420
    42         valor_SIPI_mean    0.001362
    22           valor_LST_max    0.000860
    25          valor_NDVI_std    0.000568
    18          valor_LST_mean    0.000547
    21        valor_LST_median    0.000527
    20           valor_LST_min    0.000412
    43          valor_SIPI_std    0.000397
    8          valor_FAPAR_min    0.000267
    7          valor_FAPAR_std    0.000167
    6         valor_FAPAR_mean    0.000164
    9       valor_FAPAR_median    0.000144
    17         valor_LST_count    0.000065
    31          valor_NDWI_std    0.000020
    19           valor_LST_std    0.000019
    38          valor_SAVI_min    0.000000
    11       valor_GNDVI_count    0.000000
    12        valor_GNDVI_mean    0.000000
    13         valor_GNDVI_std    0.000000
    14         valor_GNDVI_min    0.000000
    41        valor_SIPI_count    0.000000
    40          valor_SAVI_max    0.000000
    39       valor_SAVI_median    0.000000
    37          valor_SAVI_std    0.000000
    10         valor_FAPAR_max    0.000000
    36         valor_SAVI_mean    0.000000
    35        valor_SAVI_count    0.000000
    5        valor_FAPAR_count    0.000000
    15      valor_GNDVI_median    0.000000
    16         valor_GNDVI_max    0.000000
    29        valor_NDWI_count    0.000000
    23        valor_NDVI_count    0.000000



```python
metrics(test_set['target'], test_set['y_pred'])
```




    {'accuracy': 0.7869720211731662,
     'mse': 0.20792583303927603,
     'mae': 0.30645747669740453}



# Conclusiones y cambios con respecto a la rama main

- Estructura del repositorio: 
    - Create_spine.py: Editado para añadir las columnas de campaña y estado mayoritario también al spine
    - Features.py: Añadidas algunas features extra y funciones para calcularlas
    - Metrics.py: MAE incluido dentro de las métricas de evaluación de nuestros modelos
    - Models.py: Baseline + Tres modelos: Regresión logística, Gradient Boosting Classifier y Gradient Boosting Regressor

- Conclusiones del modelado:
    - Obtención de dos modelos que con parámetros iterados superan con éxito al baseline

- Dudas y next steps:
    - Probar a modelar con diferentes columnas 
    - Eliminación de nulos: Tanto al añadir las columnas de dias en estado actual/previo e indices meteorológicos perdemos muchos valores al eliminar los nulos
    - Variables categóricas: Incluir de alguna manera en nuestro modelo


