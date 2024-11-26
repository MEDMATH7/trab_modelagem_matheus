import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

class ReactorSimulator:
    def __init__(self, params):
        """
        
        Iniciar o simulador do reator com parametros
        
        Parametros:
            params: Dicionario para ser adicionado em todas questoes
        """
        #  Parametros
        self.Fi = params.get('Fi', 40)  # lbm/h
        self.V = params.get('V', 200)    # ft^3
        self.rho = params.get('rho', 50) # lbm/ft^3
        self.Cp = params.get('Cp', 0.75) # BTU/lbm-R
        self.k0 = params.get('k0', 7.08e10) # 1/h
        self.Ea1 = params.get('Ea1', 30000)  # BTU/lbm
        self.Ea2 = params.get('Ea2', 31500)  # BTU/lbm
        self.R = params.get('R', 1.99)        # BTU/lbm-R
        self.DeltaH1 = params.get('DeltaH1', 30000) # BTU/lbm
        self.DeltaH2 = params.get('DeltaH2', 15000) # BTU/lbm
        self.U = params.get('U', 150)        # BTU/h-ft^2-R
        self.A = params.get('A', 250)        # ft^2
        self.UA = self.U * self.A             # BTU/h-R
        
        # Condicao inicial
        self.CA0 = params.get('CA0', 0.1315) # lbm/ft^3
        self.CB0 = params.get('CB0', 0)
        self.CC0 = params.get('CC0', 0)
        self.T0 = params.get('T0', 540)       # R
        self.Tc0 = params.get('Tc0', 580)     # R
        
        # Condicao de entrada
        self.CA_in = params.get('CA_in', 0.2000) # lbm/ft^3
        self.T_in = params.get('T_in', 560)       # R
        self.Tc_in = params.get('Tc_in', 580)     # R
        
        # Parametros da camisa de resfriamento
        self.Fc = params.get('Fc', 40)           # lbm/h
        self.Vc = params.get('Vc', 400)          # ft^3
        
        # tempo
        self.t_span = params.get('t_span', (0, 50))  # hora
        self.t_eval = params.get('t_eval', np.linspace(0, 50, 500))  # hora -> intervalo de tempo para ser avalaido
        
        #  Pertubacoes
        self.pertubacoes = {}


    pass 
