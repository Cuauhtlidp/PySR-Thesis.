from pysr import PySRRegressor 
import re 
import os 
import sympy 
import numpy as np
import pandas as pd 
import tempfile

try:
    num_cores = os.environ["OMP_NUM_THREADS"]
except KeyError:
    from multiprocessing import cpu_count
    num_cores = cpu_count()


tmpdir = tempfile.gettempdir()
warmup_time_in_minutes = 10
custom_operators = [
    "slog(x::T) where {T} = (x > 0) ? log(x) : T(-1e9)", 
    "ssqrt(x::T) where {T} = (x > 0) ? sqrt(x) : T(-1e9)",
]

standard_operators = [
    "square",
    "cube",
    "cos", 
    "sin",
    "exp",
]

def slog_sympy(x):
    return sympy.Piecewise((sympy.log(x), x > 0), (-1e9, True))

def ssqrt_sympy(x):
    return sympy.Piecewise((sympy.sqrt(x), x > 0), (-1e9, True))

def make_est():
    return PySRRegressor(
        model_selection = "best", 
        elementwise_loss = "L1DistLoss()",
        procs = num_cores, 
        progress = False, 
        update = False, 
        precision = 64, 
        input_stream = "devnull",
        binary_operators = ["+", "-", "*", "/"],
        unary_operators = custom_operators + standard_operators, 
        maxsize = 30, 
        maxdepth = 20, 
        niterations = 1000000, 
        timeout_in_seconds = 60*(60 - warmup_time_in_minutes),
        warmup_maxsize_by = 0.2 * 0.01, 
        warm_start = False,
        tempdir = tmpdir,
        constraints = {
            "square" : 9, 
            "cube" : 9, 
            "exp" : 9,
            "sin" : 9,
            "cos" : 9, 
            "slog" : 9, 
            "ssqrt" : 5, 
            "square" : 9, 
            "cube" : 9, 
            "/" : [-1, 9], 
            "*" : [-1,-1], 
            "+" : [-1,-1],
            "-" : [-1,-1],
        }, 
        nested_constraints = {
            "cos": {"cos":0, "sin":0, "slog":0, "exp": 0}, 
            "sin": {"sin":0, "cos":0, "slog":0, "exp": 0}, 
            "/": {"/":1},
            "exp": {"exp": 0, "slog": 1, "ssqrt": 0, "sin": 0, "cos": 0}, 
            "square": {"square": 1, "cube": 1}, 
            "cube": {"cube": 1, "square": 1}, 
            "slog": {"slog": 0, "exp": 0}, 
            "ssqrt": {"ssqrt": 1, "exp": 1}, 
        },
        extra_sympy_mappings = {
            "slog" : slog_sympy,
            "ssqrt" : ssqrt_sympy,
        },
    )

def find_parens(s):
    toret = {}
    pstack = []
    for i, c in enumerate(s):
        if c == "(":
            pstack.append(i)
        elif c == ")":
            if len(pstack) == 0: 
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i
    if len(pstack) > 0: 
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))
    return toret 

def replace_prefix_operator_with_postfix(s, prefix, postfix_replacement):
    while re.search(prefix, s):
        parens_map = find_parens(s)
        start_model_str = re.search(prefix, s).span()[0]
        start_parens = re.search(prefix, s).span()[1]
        end_parens = parens_map[start_parens]
        s = (
            s[:start_model_str]
            + "("
            + s[start_parens : end_parens + 1]
            + postfix_replacement
            + ")"
            + s[end_parens + 1 :]
        )
    return s

def model(est, X=None):
    """
    Return a sympy-compatible string of the final model.
    
    Parameters
    ----------
    est : sklearn regressor
        The fitted model.
    X : pd.DataFrame, default = None
        The training data. This argument can be dropped if desired.

    Returns
    -------
    A sympy-compatible string of the final model.

    """
    model_str = est.get_best().equation
   

    return model_str

def clean_model_str(model_str):
    model_str = re.sub("slog", "log", model_str)
    model_str = re.sub("ssqrt", "sqrt", model_str)
    model_str = replace_prefix_operator_with_postfix(model_str, "square", "**2")
    model_str = replace_prefix_operator_with_postfix(model_str, "cube", "**3")
    return model_str

def models(est, X=None):
    """
    Return the pareto front of sympy-compatible strings.
    
    Parameters
    ----------
    est: sklearn regressor
        The fitted model.
    X: pd.DataFrame, default = None
        The training data. This argument can be dropped if desired.
    Returns
    -------
    A list of sympy-compatible strings of the final model.
    """
    model_strs = est.equations_.equation 
    model_strs = [clean_model_str(model_str) for model_str in model_strs]
    return model_strs


