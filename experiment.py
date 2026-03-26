from regressor import make_est
from sympy import latex
from pmlb import fetch_data 
from pathlib import Path
import pandas as pd
import numpy as np


def add_noise(dataset, rng, gamma = 0.05):   
    """
    Add noise to the target variable of the a dataset.
    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to which noise will be added. It is assumed that the target variable is named "target".
    gamma: float, default = 0.05
        
    """
    df = dataset.copy()
    target = df["target"]
    gamma_noise = gamma * np.sqrt(np.mean(np.square(target)))
    noise = rng.normal(0, gamma_noise, size = target.shape[0])
    df["target"] = target + noise
    return df

def add_noisy_features(dataset, rng):
    """
    Add noise features to a dataset.
    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to which noise features will be added. It is assumed that the target variable is named "target".
    Returns
    -------
    A new dataset with noise features added.
    """
    df = dataset.copy()
    num_rows = df.shape[0]
    num_noise_features = 3
    for i in range(1, num_noise_features + 1):
        noise_feature = rng.normal(0, 1, size = num_rows)
        df[f"noi{i}"] = noise_feature
    return df

def experiment(datasets, gamma = 0.05):
    archivo = Path("resultados.csv")
    columns = ["num_exp",
               "id_tratamiento",
               "run_id",
               "dataset",
               "num_predictors",
               "num_rows", 
               "noisy_variables",
               "noise_added", 
               "model_latex",
               "model_sympy",
               "simplified_model_latex",
               "simplified_model_sympy",
               "error", 
               "type_error",
               "error_message"]
    if archivo.is_file():
        records = pd.read_csv(archivo)
        num_exp = int(records["num_exp"].max()) + 1
    else:
        records = pd.DataFrame(columns=columns)
        num_exp = 0
    current_eq = None
    while num_exp < 288:
        est_local = make_est()
        rng = np.random.default_rng(42 + num_exp)


        num_eq = num_exp // 24  # 8 treatments × 3 runs = 24 experiments per equation
        if num_eq >= len(datasets):
            print("Todos los experimentos han sido realizados.")
            break
        if current_eq != num_eq:
            dataset = fetch_data(datasets[num_eq])
            name_dataset = datasets[num_eq]
            big_dataset = dataset.sample(n=1000, random_state=42)
            small_dataset = dataset.sample(n=100, random_state=42)
            current_eq = num_eq

        num_exp_per_equation = num_exp % 24
        id_tratamiento = num_exp_per_equation // 3
        run_id = num_exp_per_equation % 3
        if num_exp_per_equation < 12:
            dataset_in_use = big_dataset
        else:
            dataset_in_use = small_dataset
        print(f"Experimento {num_exp}/287 | dataset={name_dataset} | tratamiento={id_tratamiento} | run={run_id}")

        match num_exp_per_equation:
            case 0 | 1 | 2 | 12 | 13 | 14:   
                dataset_model = dataset_in_use
                noise = False
                noisy_variables = False
                num_features = dataset_in_use.drop(columns = ["target"]).shape[1]
            case 3 | 4 | 5 | 15 | 16 | 17:
                dataset_model = add_noise(dataset_in_use, rng, gamma)
                noise = True
                noisy_variables = False
                num_features = dataset_in_use.drop(columns = ["target"]).shape[1]
            case 6 | 7 | 8 | 18 | 19 | 20:
                dataset_model = add_noisy_features(dataset_in_use, rng)
                noise = False
                noisy_variables = True
                num_features = dataset_in_use.drop(columns = ["target"]).shape[1]
            case 9 | 10 | 11 | 21 | 22 | 23:
                dataset_model = add_noise(dataset_in_use, rng, gamma)
                dataset_model = add_noisy_features(dataset_model, rng)
                noise = True
                noisy_variables = True
                num_features = dataset_in_use.drop(columns = ["target"]).shape[1]
        X = dataset_model.drop(columns = ["target"])
        y = dataset_model["target"]
        num_rows = X.shape[0]
        try:
            est_local.fit(X, y)
            print("fit terminado", flush=True)
            expr = est_local.sympy()
            print("modelo sympy terminado", flush=True)
            model_sympy = str(expr)
            print("modelo latex terminado", flush=True)
            model_latex = latex(expr)
            simplified_model_latex = None
            simplified_model_sympy = None
            error = False 
            type_error = None
            error_message = None
        except Exception as e:
            model_latex = None
            model_sympy = None
            simplified_model_latex = None
            simplified_model_sympy = None
            error = True 
            type_error = type(e).__name__
            error_message = str(e)
        records.loc[len(records)] = [num_exp,
                                    id_tratamiento,
                                    run_id,
                                    name_dataset,
                                    num_features,
                                    num_rows,
                                    noisy_variables,
                                    noise, 
                                    model_latex,
                                    model_sympy,
                                    simplified_model_latex,
                                    simplified_model_sympy,
                                    error,
                                    type_error,
                                    error_message]
        records.to_csv(archivo, index = False) 
        print("se ha guardado exitosamente en resultados.csv", flush=True)
        
        
        num_exp += 1


    print("Experimento finalizado. Resultados guardados en resultados.csv")
        





datasets = ["feynman_I_25_13", 
            "feynman_I_29_4",
            "feynman_III_12_43",
            "feynman_I_39_1",
            "feynman_II_4_23", 
            "feynman_II_34_2",
            "feynman_III_7_38", 
            "feynman_I_34_14", 
            "feynman_test_3",
            "feynman_I_18_14", 
            "feynman_I_43_43", 
            "feynman_I_24_6"]


experiment(datasets, gamma = 0.05)






"""
archivo = Path("resultados.csv")
if archivo.is_file():
    df = pd.read_csv(archivo)
    num_exp = df["num_exp"].max()
else:
    pass




dataset = fetch_data(datasets[0]).sample(n = 1000, random_state = 42) 
X = dataset.drop(columns = ["target"])
y = dataset["target"]
est.fit(X, y)
print(est.latex())
print(latex(simplify(est.sympy())))
print(model(est))
diccionario = {"latex": est.latex(), "sympy": est.sympy(), "model": model(est)} 
df = pd.DataFrame(diccionario, index = [0])
df.to_csv("model.csv", index = False)

"""
