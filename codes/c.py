
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


solution_b = simulador.simulate()


simulador.plot_temperatures(solution_b,save_as="plot_c.png")


steady_state_b = simulador.get_steady_state(solution_b)


df = pd.DataFrame({
    'Parametros': ['CA_ss', 'CB_ss', 'CC_ss', 'T_ss', 'Tc_ss'],
    'valores em estado estacionario': [
        f"{steady_state_b['CA_ss']:.4f} lbm/ft^3",
        f"{steady_state_b['CB_ss']:.4f} lbm/ft^3",
        f"{steady_state_b['CC_ss']:.4f} lbm/ft^3",
        f"{steady_state_b['T_ss']:.2f} R",
        f"{steady_state_b['Tc_ss']:.2f} R"
    ]
})

print(df)