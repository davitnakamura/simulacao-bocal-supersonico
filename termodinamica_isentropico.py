# -*- coding: utf-8 -*-

## definnindo nome do documento
""" termodinamica_isentropico.py """

## importando bibliotecas a serem usadas
import numpy as np
from scipy.optimize import root_scalar

#importando a primeira parte (geometria)
from geometria import criar_malha, localizar_garganta

##=============================================================================
## DEFININDO EQUAÇÃO ÁREA-MACH
##=============================================================================


def equacao_area_mach (M, A_ratio, gamma=1.4):
    """ para encontrar um valor de forma que o python consiga resolver; 
    deixaremos a formulação f(M)=0 para a relação área-mach
    M: número de Mach (escalar)
        A_ratio: razão de área local A(x) / A*
        gamma: razão de calores específicos (padrão ar: 1.4)
        
    Retorno:
        f(M) que deve ser igual a zero.
    """
   
    ## separando a equação em partes para que possamos resolver posteriormente
    termo1 = 1.0/M
    termo2 = (2.0 / (gamma + 1.0)) * (1.0 + ((gamma - 1.0) / 2.0) * M**2)
    expoente = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    
    return termo1 * (termo2 ** expoente) - A_ratio


# Colocamos a definição f(M)=0 por que se tentarmos isolar M, cairíamos em uma função de grau muito alto, que não teriamos uma solução algébrica exata. se fizermos f(M)=0 podemos forçar uma situação onde encontraremos um valor específico para M; assim poderemos analizar se está supersonico, sonico ou subsonico.
# Até aqui definimos a função separando e partes para que sua escrita e a definição fique mais simplificada e fácil de executar.



 
# =============================================================================
## RESOLVER MACH BUSCANDO AS RAÍZES
# =============================================================================

def resolver_mach(A_ratio, subsonico=True, gamma=1.4):
    """
    Resolve o número de Mach para uma dada razão de áreas (A/A*).
    
    Parâmetros:
        A_ratio: razão de área A / A* (deve ser >= 1.0)
        subsonico: True para ramo subsónico (M <= 1), False para supersónico (M >= 1)
        gamma: razão de calores específicos
        
    Retorno:
        M: número de Mach correspondente.
    """
    
    
#Precisão numérica se A_ratio for menor que 1.0 - arredondamento
    if A_ratio < 1.0:
        if np.isclose(A_ratio, 1.0, atol=1e-5):
            A_ratio = 1.0
        else:
            raise ValueError(f"A razão de áreas A/A* ({A_ratio:.6f}) não pode ser menor que 1.0.")
            
# Para que A/A* não seja menor do que um (já que essa afirmação seria incorreta), arrendondaremos o valor de A/A* para 1 quando o valor for uma diferença de 10e-5, o código corrije altomaticamente para 1

 

# Para Mach = 1 (Garganta)
    if np.isclose(A_ratio, 1.0, atol=1e-4):
        return 1.0

#Definição dos intervalos para buscar as raízes
    intervalo = [1e-5, 1.0] if subsonico else [1.0, 10.0]
    
#Busca de raiz via método Brentq 
    res = root_scalar(equacao_area_mach, args=(A_ratio, gamma), bracket=intervalo, method='brentq')
    
    return res.root

 # FUNCIONAMENTO DO MÉTODO DE BRENTQ
 #


# =============================================================================
## 3. RELAÇÕES ISENTRÓPICAS E PROPRIEDADES DO ESCOAMENTO
# =============================================================================

def propriedades_isentropicas(M, p0=500000.0, T0=300.0, gamma=1.4, R=287.0):
    """
    Calcula as propriedades estáticas adimensionais e dimensionais a partir do Mach.
    
    Parâmetros:
        M: número de Mach (pode ser escalar ou vetor numpy)
        p0: pressão de estagnação (Pa)
        T0: temperatura de estagnação (K)
        gamma: razão de calores específicos
        R: constante do gás [J/(kg·K)]
        
    Retorno:
        p_p0: razão de pressão p/p0
        T_T0: razão de temperatura T/T0
        rho_rho0: razão de densidade rho/rho0
        p: pressão estática (Pa)
        T: temperatura estática (K)
        u: velocidade do escoamento (m/s)
    """
    # Razões adimensionais isentrópicas
    p_p0 = (1.0 + ((gamma - 1.0) / 2.0) * M**2) ** (-gamma / (gamma - 1.0))
    T_T0 = (1.0 + ((gamma - 1.0) / 2.0) * M**2) ** (-1.0)
    rho_rho0 = p_p0 ** (1.0 / gamma)
    
    # Propriedades dimensionais
    p = p_p0 * p0
    T = T_T0 * T0
    
    # Velocidade do som local e velocidade do escoamento
    a = np.sqrt(gamma * R * T)
    u = M * a
    
    return p_p0, T_T0, rho_rho0, p, T, u


# =============================================================================
## 4. VAZÃO MÁSSICA BLOQUEADA (CHOKED FLOW)
# =============================================================================

def calcular_vazao_maxima(A_garganta, p0=500000.0, T0=300.0, gamma=1.4, R=287.0):
    """
    Calcula a vazão mássica máxima m_dot_max (kg/s) quando o bocal está bloqueado (M=1 na garganta).
    
    Parâmetros:
        A_garganta: área da garganta A* (m²)
        p0: pressão de estagnação (Pa)
        T0: temperatura de estagnação (K)
        gamma: razão de calores específicos
        R: constante do gás [J/(kg·K)]
        
    Retorno:
        m_dot_max: vazão mássica máxima (kg/s)
    """
    termo_gama = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    m_dot_max = (p0 * A_garganta / np.sqrt(T0)) * np.sqrt(gamma / R) * termo_gama
    
    return m_dot_max


# =============================================================================
## 5. CONSTRUÇÃO DO PERFIL ISENTRÓPICO COMPLETO
# =============================================================================

def resolver_perfil_isentropico(x, A, A_garganta, x_garganta, subsonico_divergente=True, 
                                p0=500000.0, T0=300.0, gamma=1.4, R=287.0):
    """
    Varre a malha x calculando o perfil isentrópico completo de Mach e propriedades estáticas.
    
    Parâmetros:
        x: vetor de posições ao longo do bocal (m)
        A: vetor de áreas transversais (m²)
        A_garganta: área mínima A* (m²)
        x_garganta: posição x onde ocorre a garganta (m)
        subsonico_divergente: True para escoamento subsónico no divergente, 
                              False para escoamento supersónico no divergente.
    """
    N = len(x)
    M = np.zeros(N)
    
    # Encontra o índice na malha que está mais próximo da posição real da garganta
    idx_garganta = np.argmin(np.abs(x - x_garganta))
    
    for i in range(N):
        # Na garganta exata da malha, impõe Mach = 1.0
        if i == idx_garganta:
            M[i] = 1.0
            continue
            
        A_ratio = A[i] / A_garganta
        
        # Convergente (x < x_garganta): sempre subsónico
        if x[i] < x_garganta:
            M[i] = resolver_mach(A_ratio, subsonico=True, gamma=gamma)
        # Divergente (x > x_garganta): depende do regime escolhido
        else:
            M[i] = resolver_mach(A_ratio, subsonico=subsonico_divergente, gamma=gamma)
            
    p_p0, T_T0, rho_rho0, p, T, u = propriedades_isentropicas(M, p0, T0, gamma, R)
    
    return M, p_p0, T_T0, rho_rho0, p, T, u


# =============================================================================
## BLOCO DE TESTE DA PARTE 2
# =============================================================================

if __name__ == "__main__":
    
    print("====================================================")
    print("      TESTE DA PARTE 2: MÓDULO TERMODINÂMICO        ")
    print("====================================================\n")

    # 1. Condições de entrada do problema
    gamma = 1.4
    R = 287.0        # J/(kg·K)
    p0 = 500000.0    # 5 bar = 500 kPa
    T0 = 300.0       # K

    # 2. Obtendo a geometria do módulo da Parte 1 (geometria.py)
    x, A = criar_malha(iniciocomprimento=0.0, totalcomprimento=0.1, numerodepontos=500)
    x_garganta, A_garganta, _ = localizar_garganta(x, A)

    # 3. Teste da Vazão Mássica Bloqueada
    m_dot_max = calcular_vazao_maxima(A_garganta, p0, T0, gamma, R)
    print("--- 1. VAZÃO MÁSSICA BLOQUEADA ---")
    print(f"Área da garganta (A*): {A_garganta:.8e} m²")
    print(f"Vazão mássica máx (m_dot): {m_dot_max:.6f} kg/s\n")

    # 4. Teste do Perfil Totalmente Subsónico (Isentrópico)
    M_sub, p_p0_sub, T_T0_sub, _, p_sub, _, u_sub = resolver_perfil_isentropico(
        x, A, A_garganta, x_garganta, subsonico_divergente=True, p0=p0, T0=T0, gamma=gamma, R=R
    )

    print("--- 2. PERFIL TOTALMENTE SUBSÓNICO (ISENTRÓPICO) ---")
    print(f"Entrada  (x = {x[0]:.2f}m) -> Mach: {M_sub[0]:.4f} | p/p0: {p_p0_sub[0]:.4f} | p: {p_sub[0]/1e5:.4f} bar | u: {u_sub[0]:.2f} m/s")
    
    idx_garganta = np.argmin(np.abs(x - x_garganta))
    print(f"Garganta (x = {x[idx_garganta]:.4f}m) -> Mach: {M_sub[idx_garganta]:.4f} | p/p0: {p_p0_sub[idx_garganta]:.4f} | p: {p_sub[idx_garganta]/1e5:.4f} bar | u: {u_sub[idx_garganta]:.2f} m/s")
    
    print(f"Saída    (x = {x[-1]:.2f}m) -> Mach: {M_sub[-1]:.4f} | p/p0: {p_p0_sub[-1]:.4f} | p: {p_sub[-1]/1e5:.4f} bar | u: {u_sub[-1]:.2f} m/s\n")

    # 5. Teste do Perfil Supersónico no Divergente (Condição de Projeto Isentrópica)
    M_sup, p_p0_sup, T_T0_sup, _, p_sup, _, u_sup = resolver_perfil_isentropico(
        x, A, A_garganta, x_garganta, subsonico_divergente=False, p0=p0, T0=T0, gamma=gamma, R=R
    )

    print("--- 3. PERFIL SUPERSÓNICO NO DIVERGENTE (CONDIÇÃO DE PROJETO) ---")
    print(f"Garganta (x = {x[idx_garganta]:.4f}m) -> Mach: {M_sup[idx_garganta]:.4f} | p/p0: {p_p0_sup[idx_garganta]:.4f}")
    print(f"Saída    (x = {x[-1]:.2f}m) -> Mach: {M_sup[-1]:.4f} | p/p0: {p_p0_sup[-1]:.4f} | p: {p_sup[-1]/1e5:.4f} bar | u: {u_sup[-1]:.2f} m/s")