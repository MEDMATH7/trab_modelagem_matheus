import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

class ReactorSimulator:
    def __init__(self, params):
        """
        Iniciar o simulador do reator com parâmetros

        Parâmetros:
            params: Dicionário contendo os parâmetros do reator
        """
        # Parâmetros do reator com valores padrão
        self.Fi = params.get('Fi', 40)  # Taxa de fluxo de entrada (lbm/h)
        self.V = params.get('V', 200)    # Volume do reator (ft^3)
        self.rho = params.get('rho', 50) # Densidade (lbm/ft^3)
        self.Cp = params.get('Cp', 0.75) # Calor específico (BTU/lbm-R)
        self.k0 = params.get('k0', 7.08e10) # Fator pré-exponencial (1/h)
        self.Ea1 = params.get('Ea1', 30000)  # Energia de ativação da reação 1 (BTU/lbm)
        self.Ea2 = params.get('Ea2', 31500)  # Energia de ativação da reação 2 (BTU/lbm)
        self.R = params.get('R', 1.99)        # Constante dos gases (BTU/lbm-R)
        self.DeltaH1 = params.get('DeltaH1', 30000) # Variação de entalpia da reação 1 (BTU/lbm)
        self.DeltaH2 = params.get('DeltaH2', 15000) # Variação de entalpia da reação 2 (BTU/lbm)
        self.U = params.get('U', 150)        # Coeficiente de transferência de calor (BTU/h-ft^2-R)
        self.A = params.get('A', 250)        # Área de superfície para troca de calor (ft^2)
        self.UA = self.U * self.A             # Produto UA (BTU/h-R)

        # Condição inicial do reator
        self.CA0 = params.get('CA0', 0.1315) # Concentração inicial de A (lbm/ft^3)
        self.CB0 = params.get('CB0', 0)      # Concentração inicial de B (lbm/ft^3)
        self.CC0 = params.get('CC0', 0)      # Concentração inicial de C (lbm/ft^3)
        self.T0 = params.get('T0', 540)       # Temperatura inicial do reator (R)
        self.Tc0 = params.get('Tc0', 580)     # Temperatura inicial da camisa de resfriamento (R)

        # Condição de entrada
        self.CA_in = params.get('CA_in', 0.2000) # Concentração de A na entrada (lbm/ft^3)
        self.T_in = params.get('T_in', 560)       # Temperatura de entrada (R)
        self.Tc_in = params.get('Tc_in', 580)     # Temperatura de entrada da camisa de resfriamento (R)

        # Parâmetros da camisa de resfriamento
        self.Fc = params.get('Fc', 40)           # Taxa de fluxo de resfriamento (lbm/h)
        self.Vc = params.get('Vc', 400)          # Volume da camisa de resfriamento (ft^3)

        # Intervalo de tempo para a simulação
        self.t_span = params.get('t_span', (0, 50)) 
        self.t_eval = params.get('t_eval', np.linspace(0, 50, 500))

        # Perturbações (inicialmente vazias)
        self.pertubacoes = {}

    def k1(self, T):
        """
        Calcula a constante de taxa k1 baseada na temperatura T
        """
        return self.k0 * np.exp(-self.Ea1 / (self.R * T))
    
    def k2(self, T):
        """
        Calcula a constante de taxa k2 baseada na temperatura T
        """
        return self.k0 * np.exp(-self.Ea2 / (self.R * T))

    def simulate(self, pertubacoes=None, t_span=None, t_eval=None, y0=None):
        """
        Executa a simulação do reator resolvendo as equações diferenciais
        """
        # Utiliza os parâmetros padrão se não forem fornecidos
        if t_span is None:
            t_span = self.t_span
        if t_eval is None:
            t_eval = self.t_eval
        if y0 is None:
            y0 = [self.CA0, self.CB0, self.CC0, self.T0, self.Tc0]
        
        def reactor_odes(t, y):
            """
            Define as equações diferenciais para o reator
            """
            CA, CB, CC, T, Tc = y
            CA_in_current = self.CA_in
            Tc_in_current = self.Tc_in

            # Aplica perturbações se existirem
            if pertubacoes is not None:
                if 'CA_in' in pertubacoes and t >= pertubacoes['CA_in']['time']:
                    CA_in_current = pertubacoes['CA_in']['value']
                if 'Tc_in' in pertubacoes and t >= pertubacoes['Tc_in']['time']:
                    Tc_in_current = pertubacoes['Tc_in']['value']

            # Calcula as taxas de reação
            r1 = self.k1(T) * CA
            r2 = self.k2(T) * CB
         
            # Equações diferenciais para as concentrações
            dCA_dt = (self.Fi / self.V) * (CA_in_current - CA) - r1
            dCB_dt = - (self.Fi / self.V) * CB + r1 - r2
            dCC_dt = - (self.Fi / self.V) * CC + r2
         
            # Equação diferencial para a temperatura do reator
            dT_dt = (self.Fi * self.rho * self.Cp * (self.T_in - T) +
                     (-self.DeltaH1) * self.V * r1 +
                     (-self.DeltaH2) * self.V * r2 -
                     self.UA * (T - Tc)) / (self.rho * self.V * self.Cp)
            
            # Equação diferencial para a temperatura da camisa de resfriamento
            dTc_dt = (self.UA * (T - Tc) + self.Fc * self.rho * self.Cp * (Tc_in_current - Tc)) / (self.rho * self.V * self.Cp)
            
            return [dCA_dt, dCB_dt, dCC_dt, dT_dt, dTc_dt]
        
        # Resolve as equações diferenciais
        solution = solve_ivp(reactor_odes, t_span, y0, t_eval=t_eval)
        return solution

    def plot_concentrations(self, solution, save_as=None):
        """
        Plota os gráficos de concentração das espécies A, B e C
        """
        CA, CB, CC = solution.y[0], solution.y[1], solution.y[2]
        plt.figure(figsize=(10,6))
        plt.plot(solution.t, CA, label='CA')
        plt.plot(solution.t, CB, label='CB')
        plt.plot(solution.t, CC, label='CC')
        plt.xlabel('Tempo (h)')
        plt.ylabel('Concentração (lbm/ft^3)')
        plt.legend()
        plt.title('Perfis de Concentração')
        plt.grid(True)
        if save_as:
            plt.savefig(save_as, dpi=300)
            print(f"Gráfico salvo como {save_as}")
        else:
            plt.show()

    def plot_temperatures(self, solution, save_as=None):
        """
        Plota os gráficos de temperatura do reator e da camisa de resfriamento
        """
        T, Tc = solution.y[3], solution.y[4]
        plt.figure(figsize=(10,6))
        plt.plot(solution.t, T, label='Temperatura do Reator')
        plt.plot(solution.t, Tc, label='Temperatura da Camisa de Resfriamento')
        plt.xlabel('Tempo (h)')
        plt.ylabel('Temperatura (R)')
        plt.legend()
        plt.title('Perfis de Temperatura')
        plt.grid(True)
        if save_as:
            plt.savefig(save_as, dpi=300)
            print(f"Gráfico salvo como {save_as}")
        else:
            plt.show()

    def get_steady_state(self, solution):
        """
        Obtém o estado estacionário da simulação
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

    def simulate_disturbance(self, pertubacoes_time, pertubacoes_valores, t_end):
        """
        Simula o reator com perturbações nos parâmetros de entrada

        Parâmetros:
            pertubacoes_time: Tempo em que a perturbação ocorre
            pertubacoes_valores: Valores das perturbações
            t_end: Tempo final da simulação
        """
        t_span = (self.t_span[0], t_end)
        t_eval = np.linspace(t_span[0], t_end, int((t_end - self.t_span[0]) * 1000))

        # Define as perturbações
        pertubacoes = {}
        for var, val in pertubacoes_valores.items():
            pertubacoes[var] = {'time': pertubacoes_time, 'value': val}

        # Executa a simulação com perturbações
        solution = self.simulate(pertubacoes=pertubacoes, t_span=t_span, t_eval=t_eval)

        return solution

    def sensitivity_analysis(self, param, mud_percent):
        """
        Realiza análise de sensibilidade sobre os parâmetros especificados

        Parâmetros:
            param -> Parâmetro a ser variado ('rho_c', 'Cp_c' ou 'both')
            mud_percent (list ou array): Mudança nos valores em porcentagem (ex: [-0.5, 0.25, 1.0, ...])
        """
        data = []

        for pct in mud_percent:
            # Ajusta os parâmetros conforme a análise de sensibilidade
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
                raise ValueError("Deve variar somente 'rho_c', 'Cp_c' ou 'both'")

            # Define as equações diferenciais modificadas para a análise de sensibilidade
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

            # Resolve as equações diferenciais para os parâmetros variáveis
            solution_sens = solve_ivp(
                reactor_odes_sensitivity, 
                self.t_span, 
                [self.CA0, self.CB0, self.CC0, self.T0, self.Tc0], 
                t_eval=self.t_eval
            )
            
            # Obtém o estado estacionário da simulação sensível
            CA_ss = solution_sens.y[0, -1]
            CB_ss = solution_sens.y[1, -1]
            CC_ss = solution_sens.y[2, -1]
            T_ss = solution_sens.y[3, -1]
            Tc_ss = solution_sens.y[4, -1]
            
            # Adiciona os resultados à lista de dados
            data.append({
                '% Change': pct * 100,
                'CA_ss': CA_ss,
                'CB_ss': CB_ss,
                'CC_ss': CC_ss,
                'T_ss': T_ss,
                'Tc_ss': Tc_ss
            })

        # Cria um DataFrame a partir dos dados coletados
        results = pd.DataFrame(data)

        return results

    def plot_sensitivity(self, results, variable, save_as=None):
        """
        Plota os resultados da análise de sensibilidade para uma variável específica

        Parâmetros:
            results: DataFrame contendo os resultados da análise de sensibilidade
            variable: Nome da variável a ser plotada (ex: 'CA_ss', 'CB_ss', etc.)
            save_as: Nome do arquivo para salvar o gráfico (opcional)
        """
        plt.figure(figsize=(10,6))
        plt.plot(results['% Change'], results[variable], marker='o')
        plt.xlabel('% de Mudança em rho_c e Cp_c')
        plt.ylabel(variable)
        plt.title(f'Análise de Sensibilidade: {variable} vs % de Mudança nas Propriedades do Fluido de Resfriamento')
        plt.grid(True)
        if save_as:
            plt.savefig(save_as, dpi=300)
            print(f"Gráfico salvo como {save_as}")
        else:
            plt.show()
