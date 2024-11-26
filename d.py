# Imports especificos
from main import ReactorSimulator

# Imports gerais
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd


params = {
    'Fi': 40,            # lbm/h
    'V': 200,            # ft^3
    'rho': 50,           # lbm/ft^3
    'Cp': 0.75,          # BTU/lbm-R
    'k0': 7.08e10,       # 1/h
    'Ea1': 30000,        # BTU/lbm
    'Ea2': 31500,        # BTU/lbm
    'R': 1.99,           # BTU/lbm-R
    'DeltaH1': 30000,    # BTU/lbm
    'DeltaH2': 15000,    # BTU/lbm
    'U': 150,            # BTU/h-ft^2-R
    'A': 250,            # ft^2
    'CA0': 0.1315,       # lbm/ft^3
    'CB0': 0,            # lbm/ft^3
    'CC0': 0,            # lbm/ft^3
    'T0': 540,           # R
    'Tc0': 580,          # R
    'CA_in': 0.2000,     # lbm/ft^3
    'T_in': 560,         # R
    'Tc_in': 580,        # R
    'Fc': 40,            # lbm/h
    'Vc': 400,           # ft^3
    't_span': (0, 50),   # hora
    't_eval': np.linspace(0, 50, 500)  # hora
}


simulador = ReactorSimulator(params)

pertubacoes_time = 50
pertubacoes_valores = {'Tc_in': 600}
t_end = 100

solution_d = simulador.simulate_disturbance(pertubacoes_time,pertubacoes_valores,t_end)



simulador.plot_concentrations(solution_d,save_as="plot_d_conc.png")

simulador.plot_temperatures(solution_d,save_as="plot_d_temp.png")