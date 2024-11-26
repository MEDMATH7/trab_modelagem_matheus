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

            # quando ha pertubacoes
            def reactor_odes_disturbance(t, y):
                CA, CB, CC, T, Tc = y
                # Check for disturbances
                if 'Tc_in' in pertubacoes:
                    if t >= pertubacoes['Tc_in']['time']:
                        Tc_in_current = pertubacoes['Tc_in']['value']
                    else:
                        Tc_in_current = self.Tc_in
                else:
                    Tc_in_current = self.Tc_in
                
                if 'CA_in' in pertubacoes:
                    if t >= pertubacoes['CA_in']['time']:
                        CA_in_current = pertubacoes['CA_in']['value']
                    else:
                        CA_in_current = self.CA_in
                else:
                    CA_in_current = self.CA_in
                
                # taxa de reacao
                r1 = self.k1(T) * CA
                r2 = self.k2(T) * CB
                # BM
                dCA_dt = (self.Fi / self.V) * (CA_in_current - CA) - r1
                dCB_dt = - (self.Fi / self.V) * CB + r1 - r2
                dCC_dt = - (self.Fi / self.V) * CC + r2
                # BEEnergy balances
                dT_dt = (self.Fi * self.rho * self.Cp * (self.T_in - T) + 
                         (-self.DeltaH1) * self.V * r1 + 
                         (-self.DeltaH2) * self.V * r2 - 
                         self.UA * (T - Tc)) / (self.rho * self.V * self.Cp)
                dTc_dt = (self.UA * (T - Tc) + self.Fc * self.rho * self.Cp * (Tc_in_current - Tc)) / (self.rho * self.V * self.Cp)
                return [dCA_dt, dCB_dt, dCC_dt, dT_dt, dTc_dt]
            return solve_ivp(reactor_odes_disturbance, self.t_span, [self.CA0, self.CB0, self.CC0, self.T0, self.Tc0], t_eval=self.t_eval)

    def plot_concentrations(self, solution):
        """
        plotar os graficos de concentracao
        """
        CA, CB, CC = solution.y[0], solution.y[1], solution.y[2]
        plt.figure(figsize=(10,6))
        plt.plot(solution.t, CA, label='CA')
        plt.plot(solution.t, CB, label='CB')
        plt.plot(solution.t, CC, label='CC')
        plt.xlabel('Time (h)')
        plt.ylabel('Concentracao (lbm/ft^3)')
        plt.legend()
        plt.title('Perfis de concentracao')
        plt.grid(True)
        plt.show()
    
    def plot_temperatures(self, solution):
        """
        Plotar os graficos de temperatura
        """
        T, Tc = solution.y[3], solution.y[4]
        plt.figure(figsize=(10,6))
        plt.plot(solution.t, T, label='Temperatura do reator')
        plt.plot(solution.t, Tc, label='Temperatura da camsia de resfriamento')
        plt.xlabel('Time (h)')
        plt.ylabel('Temperatura (R)')
        plt.legend()
        plt.title('Perfis de Temperatura')
        plt.grid(True)
        plt.show()
    

    def get_steady_state(self, solution):
        """
    This function will make the estado estacionario from scipy.
        """
        CA_ss = solution.y[0, -1]
        CB_ss = solution.y[1, -1]
        CC_ss = solution.y[2, -1]
        T_ss = solution.y[3, -1]
        Tc_ss = solution.y[4, -1]
        return {
            'CA_ss': CA_ss,
            'CB_ss': CB_ss,
            'CC_ss': CC_ss,
            'T_ss': T_ss,
            'Tc_ss': Tc_ss
        }
    
    def simulate_disturbance(self, pertubacoes_time, pertubacoes_valores):
        """
        simula sob pertubacoes
        
        Parametros:
            pertubacoes_time: tempo onde temos pertubacao (hours)
            pertubacoes_valores dicionario...
        """        
        pertubacao = {}
        for var, val in pertubacoes_valores.items():
            pertubacao[var] = {'time': pertubacoes_time, 'value': val}
        solution = self.simulate(disturbance=pertubacao)
        return solution
    
    def sensitivity_analysis(self, param, mud_percent):
        """
        Parametros:
            param -> dicionario de parametros
            mud_percent (list or array): mudanca nos valores em porcentagem (e.g., [-0.5, 0.25, 1.0, ...])
        """
        data = []

        for pct in mud_percent:
            if param == 'rho_c':
                rho_c_var = self.rho * (1 + pct)
                Cp_c_var = self.Cp
            elif param == 'Cp_c':
                rho_c_var = self.rho
                Cp_c_var = self.Cp * (1 + pct)
            elif param == 'both':
                rho_c_var = self.rho * (1 + pct)
                Cp_c_var = self.Cp * (1 + pct)
            else:
                raise ValueError("deve variar somente 'rho_c', 'Cp_c' ou 'both'")

            # modifica as equacoes ordinarias
            def reactor_odes_sensitivity(t, y):
                CA, CB, CC, T, Tc = y
                
                r1 = self.k1(T) * CA
                r2 = self.k2(T) * CB
                
                dCA_dt = (self.Fi / self.V) * (self.CA_in - CA) - r1
                dCB_dt = - (self.Fi / self.V) * CB + r1 - r2
                dCC_dt = - (self.Fi / self.V) * CC + r2
            
                dT_dt = (self.Fi * self.rho * self.Cp * (self.T_in - T) +
                         (-self.DeltaH1) * self.V * r1 +
                         (-self.DeltaH2) * self.V * r2 -
                         self.UA * (T - Tc)) / (self.rho * self.V * self.Cp)
                dTc_dt = (self.UA * (T - Tc) + self.Fc * rho_c_var * Cp_c_var * (self.Tc_in - Tc)) / (rho_c_var * self.V * Cp_c_var)
                return [dCA_dt, dCB_dt, dCC_dt, dT_dt, dTc_dt]

            
            solution_sens = solve_ivp(reactor_odes_sensitivity, self.t_span, [self.CA0, self.CB0, self.CC0, self.T0, self.Tc0], t_eval=self.t_eval)
            
            CA_ss = solution_sens.y[0, -1]
            CB_ss = solution_sens.y[1, -1]
            CC_ss = solution_sens.y[2, -1]
            T_ss = solution_sens.y[3, -1]
            Tc_ss = solution_sens.y[4, -1]
            
            data.append({
                '% Change': pct * 100,
                'CA_ss': CA_ss,
                'CB_ss': CB_ss,
                'CC_ss': CC_ss,
                'T_ss': T_ss,
                'Tc_ss': Tc_ss
            })

        # Create DataFrame from collected data
        results = pd.DataFrame(data)

        return 
    
    def plot_sensitivity(self, results, variable):
        """
        Plot sensitivity analysis results.
        """
        plt.figure(figsize=(10,6))
        plt.plot(results['% Change'], results[variable], marker='o')
        plt.xlabel('% Change in rho_c and Cp_c')
        plt.ylabel(variable)
        plt.title(f'Sensitivity Analysis: {variable} vs % Change in Cooling Fluid Properties')
        plt.grid(True)
        plt.show()


