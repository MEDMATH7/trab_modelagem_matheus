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

mud_percent = np.arange(-0.5, 4.25, 0.25)

sensitivity_results = simulador.sensitivity_analysis(param='both', mud_percent=mud_percent)

print(sensitivity_results)


plt.figure(figsize=(10,6))
plt.plot(sensitivity_results['% Change'], sensitivity_results['T_ss'], marker='o')
plt.xlabel('% mudanca em rho_c e Cp_c')
plt.ylabel('Estado estacionario no Reator, Temperatura (R)')
plt.title('Analise de sensibilidade: Temp reator vs % Mudanca nas propriedades do fluido de resfriamento')
plt.grid(True)

plt.savefig("plot_f_mudanca_rhoc_cpc_reator", dpi=300)


# Plotting steady-state cooling jacket temperature vs % Change
plt.figure(figsize=(10,6))
plt.plot(sensitivity_results['% Change'], sensitivity_results['Tc_ss'], marker='o', color='orange')
plt.xlabel('% mudanca em rho_c e Cp_c')
plt.ylabel('Estado estacionario na camisa de resfriamento, Temperatura (R)')
plt.title('Analise de sensibilidade: camisa de resfriamento vs % Mudanca nas propriedades do fluido de resfriamento')
plt.grid(True)
plt.savefig("plot_f_mudanca_rhoc_cpc_camisa", dpi=300)