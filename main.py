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


    def k1(self, T):
        """
        Calcula k1
        """
        return self.k0 * np.exp(-self.Ea1 / (self.R * T))
    
    def k2(self, T):
        """
        Calcula k2
        """
        return self.k0 * np.exp(-self.Ea2 / (self.R * T))
    
    def reactor_odes(self, t, y):
        """
        criando as equacoes diferenciais
        
        parametros:
            t: tempo
            y: CA, CB, CC, T, Tc
        
        gera uma lista de variaveis [dCA/dt, dCB/dt, dCC/dt, dT/dt, dTc/dt]
        """
        CA, CB, CC, T, Tc = y
        # Reacao
        r1 = self.k1(T) * CA
        r2 = self.k2(T) * CB
        # Balanco de massa
        dCA_dt = (self.Fi / self.V) * (self.CA_in - CA) - r1
        dCB_dt = - (self.Fi / self.V) * CB + r1 - r2
        dCC_dt = - (self.Fi / self.V) * CC + r2
        # Balanco de energia
        dT_dt = (self.Fi * self.rho * self.Cp * (self.T_in - T) + 
                 (-self.DeltaH1) * self.V * r1 + 
                 (-self.DeltaH2) * self.V * r2 - 
                 self.UA * (T - Tc)) / (self.rho * self.V * self.Cp)
        dTc_dt = (self.UA * (T - Tc) + self.Fc * self.rho * self.Cp * (self.Tc_in - Tc)) / (self.rho * self.V * self.Cp)
        return [dCA_dt, dCB_dt, dCC_dt, dT_dt, dTc_dt]
    
    def simulate(self, pertubacoes=None):
        """
        com solve_ivp do scipy ai simular as equacoes ordinarias
        
        parametros:
            pertubacoes: Ha no dicionario algumas (Tc_in e CA_in). 
        """
        if pertubacoes is None:
            # sem pertubacao
            return solve_ivp(self.reactor_odes, self.t_span, [self.CA0, self.CB0, self.CC0, self.T0, self.Tc0], t_eval=self.t_eval)
        else:

            # TBD
            
            pass
        
         
