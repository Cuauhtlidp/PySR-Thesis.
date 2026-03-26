"""
Análisis general y selección de los datasets para el diseño de experimentos.

Overall analysis and selection of datasets for experimental design.
"""
from pmlb import fetch_data, dataset_names
import pandas as pd
import numpy as np

def explore_datasets_pmlb(nombres):
  headers = ["Dataset name","num. rows", "num. predictors", "correlation", "mean target", "std target", "MS Value" ]
  data_set = pd.DataFrame(columns = headers)
  for nombre in nombres:
    array = []
    array_corr = []
    dataset = fetch_data(nombre)
    numero = dataset.shape[0]
    predictores = dataset.shape[1] - 1
    dataset_sin_target = dataset.drop(columns = ["target"])
    target = dataset["target"]
    media_target = target.mean()
    std_target = target.std()
    ms_value = np.sqrt(np.sum((target)**2)/len(target)) #useful for adding noise to the target variable in the future
    corr_p = dataset_sin_target.corr(method = "pearson")
    for i in range(corr_p.shape[0]):
      for j in range(corr_p.shape[1]):
        if i == j:
          continue
        else:
          array_corr.append(np.abs(corr_p.iloc[i,j]))
    array_true = np.array(array_corr) < 0.5
    if False in array_true:
      hay_correlacion = True
    else:
      hay_correlacion = False
    array.append(nombre)
    array.append(numero)
    array.append(predictores)
    array.append(hay_correlacion)
    array.append(media_target)
    array.append(std_target)
    array.append(ms_value)
    data_set.loc[len(data_set)] = array
  return data_set

def select_datasets(data_set, max_num_predictors = 4, num_datasets = 4, random_state = 42):
    headers = data_set.columns
    final_dataset = pd.DataFrame(columns = headers)
    for num_predictors in range(1, max_num_predictors + 1):
        data_set_filtrado = data_set[data_set["num. predictors"] == num_predictors]
        if len(data_set_filtrado) >= num_datasets:
            data_set_filtrado = data_set_filtrado.sample(n = num_datasets, random_state = random_state)
            final_dataset = pd.concat([final_dataset, data_set_filtrado], ignore_index = True)
    return final_dataset
    
  
names = [d for d in dataset_names if "feynman" in d]

data_set = explore_datasets_pmlb(names)
final_dataset = select_datasets(data_set, max_num_predictors = 4, num_datasets = 4, random_state = 42)
data_set.to_csv("datasets_pmlb.csv", index = False)
final_dataset.to_csv("final_datasets_pmlb.csv", index = False)
final_dataset.to_latex("final_datasets_pmlb.tex", index = False)

