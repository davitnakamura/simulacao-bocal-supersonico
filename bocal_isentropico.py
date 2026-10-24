## Definindo o nome do documento
""" bocal_isentropico.py - 

Módulo A: Geometria, relação área-Mach, escoamento isentrópico, vazão bloqueada, pressões críticas
e classificação do regime de operação.
"""

## Importando as bibliotecas a serem usadas
import numpy as np
from scipy.optimize import root_scalar

## Constantes compartilhadas
from parametros import GAMMA, R_GAS, P0, T0, L

## Importando a primeira parte (geometria)
from geometria import area_bocal, criar_malha, localizar_garganta


##=============================================================================
## GEOMETRIA (PERFIL, DERIVADA, GARGANTA E SAÍDA)
##=============================================================================

A_perfil = area_bocal          # A(x) [m²]; aceita escalar ou vetor


def dAdx_perfil(x, h=1e-7):
    """dA/dx por diferença central (a geometria.py não fornece a derivada analítica).
    Usada apenas para conferir que dA/dx ~ 0 na garganta."""
    return (A_perfil(x + h) - A_perfil(x - h)) / (2.0 * h)


## Garganta pela condição dA/dx = 0, numa malha fina (independente da malha dos gráficos)
_x_malha, _A_malha = criar_malha(0.0, L, 2001)
X_GARGANTA, A_GARGANTA, _ = localizar_garganta(_x_malha, _A_malha)   # ~0.0580 m ; 0.9375e-4 m²
A_SAIDA = A_perfil(L)                                                 # A_e = 1.5e-4 m²


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


# Colocamos a definição f(M)=0 porque, se tentarmos isolar M, cairíamos em uma função de grau muito alto, que não teria solução algébrica exata. Se fizermos f(M)=0, podemos buscar numericamente o valor específico de M; assim poderemos analisar se o escoamento está supersônico, sônico ou subsônico.
# Até aqui definimos a função separando em partes para que sua escrita e a definição fiquem mais simples e fáceis de executar.


def razao_area(M, g=GAMMA):
    """Devolve A/A*(M). Como equacao_area_mach(M, A_ratio) = A/A*(M) - A_ratio,
    basta usar A_ratio = 0 para sobrar só A/A*(M)."""
    return equacao_area_mach(M, 0.0, g)


# =============================================================================
## RESOLVER MACH BUSCANDO AS RAÍZES
# =============================================================================

## Tolerâncias
TOL_ERRO = 1e-6        # abaixo de 1 - TOL_ERRO, A/A* < 1 é fisicamente impossível -> erro
TOL_GARGANTA = 1e-12   # se |A/A* - 1| <= TOL_GARGANTA, consideramos a garganta (M = 1)
## Perto de M = 1, A/A* varia com (M-1)², então a tolerância em A/A* precisa ser pequena:
## com 1e-4, o Mach ficaria "travado" em 1.0 numa faixa de até ~1,3% e a vazão deixaria de ser
## constante ao longo de x. Com 1e-12, o erro máximo em M é ~1e-6.

def resolver_mach(A_ratio, subsonico=True, gamma=1.4):
    """
    Resolve o número de Mach para uma dada razão de áreas (A/A*).
    
    Parâmetros:
        A_ratio: razão de área A / A* (deve ser >= 1.0)
        subsonico: True para ramo subsônico (M <= 1), False para supersônico (M >= 1)
        gamma: razão de calores específicos
        
    Retorno:
        M: número de Mach correspondente.
    """
    
    # Precisão numérica: se A_ratio for ligeiramente menor que 1.0 (erro de arredondamento),
    # o valor é tratado como 1.0; se for bem menor, a razão é fisicamente impossível.
    if A_ratio < 1.0 - TOL_ERRO:
        raise ValueError(f"A razão de áreas A/A* ({A_ratio:.6f}) não pode ser menor que 1.0.")

    # Para Mach = 1 (garganta)
    if A_ratio <= 1.0 + TOL_GARGANTA:
        return 1.0

    # Definição dos intervalos para buscar as raízes
    intervalo = [1e-5, 1.0] if subsonico else [1.0, 10.0]
    
    # Busca da raiz pelo método de Brent (brentq)
    res = root_scalar(equacao_area_mach, args=(A_ratio, gamma), bracket=intervalo, method='brentq')
    
    return res.root

# FUNCIONAMENTO DO MÉTODO DE BRENT (brentq): exige um intervalo [a, b] em que f(a) e f(b) tenham
# sinais opostos; como A/A*(M) é monótona em cada ramo, existe uma única raiz no intervalo.


def mach_de_area(r, ramo, g=GAMMA):
    """Interface usada pelos outros módulos. ramo = 'sub' ou 'sup'."""
    if ramo not in ('sub', 'sup'):
        raise ValueError("ramo deve ser 'sub' ou 'sup'")
    return resolver_mach(r, subsonico=(ramo == 'sub'), gamma=g)


# =============================================================================
## 3. RELAÇÕES ISENTRÓPICAS E PROPRIEDADES DO ESCOAMENTO
# =============================================================================

def propriedades_isentropicas(M, p0=P0, T0=T0, gamma=GAMMA, R=R_GAS):
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


def p_sobre_p0(M, g=GAMMA):
    """p/p0 isentrópico em função de M (escalar ou vetor)."""
    return propriedades_isentropicas(M, gamma=g)[0]


def T_sobre_T0(M, g=GAMMA):
    """T/T0 isentrópico em função de M (escalar ou vetor)."""
    return propriedades_isentropicas(M, gamma=g)[1]


# =============================================================================
## 4. VAZÃO MÁSSICA BLOQUEADA (CHOKED FLOW)
# =============================================================================

def calcular_vazao_maxima(A_garganta, p0=P0, T0=T0, gamma=GAMMA, R=R_GAS):
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
                                p0=P0, T0=T0, gamma=GAMMA, R=R_GAS):
    """
    Varre a malha x calculando o perfil isentrópico completo de Mach e propriedades estáticas.
    
    Parâmetros:
        x: vetor de posições ao longo do bocal (m)
        A: vetor de áreas transversais (m²)
        A_garganta: área mínima A* (m²)
        x_garganta: posição x onde ocorre a garganta (m)
        subsonico_divergente: True para escoamento subsônico no divergente, 
                              False para escoamento supersônico no divergente.

    ATENÇÃO: com subsonico_divergente=True a garganta é imposta como M = 1 (A* = A_garganta).
    Isso corresponde SÓ ao caso p_b = pI (regime b). Para p_b > pI (regime a), a garganta
    NÃO é sônica: use perfil_subsonico_total(p_b, x), mais abaixo.
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
        
        # Convergente (x < x_garganta): sempre subsônico
        if x[i] < x_garganta:
            M[i] = resolver_mach(A_ratio, subsonico=True, gamma=gamma)
        # Divergente (x > x_garganta): depende do regime escolhido
        else:
            M[i] = resolver_mach(A_ratio, subsonico=subsonico_divergente, gamma=gamma)
            
    p_p0, T_T0, rho_rho0, p, T, u = propriedades_isentropicas(M, p0, T0, gamma, R)
    
    return M, p_p0, T_T0, rho_rho0, p, T, u


# =============================================================================
## PRESSÕES CRÍTICAS (pI, pII, pIII)
# =============================================================================

def pressoes_criticas(Ae_At=A_SAIDA / A_GARGANTA, g=GAMMA):
    """Devolve (p_I, p_II, p_III) em Pa. Ordem: p_III < p_II < p_I < p0."""
    M_sub = mach_de_area(Ae_At, 'sub', g)      # saída subsônica, garganta sônica
    M_sup = mach_de_area(Ae_At, 'sup', g)      # saída supersônica (projeto)
    pI = P0 * p_sobre_p0(M_sub, g)
    pIII = P0 * p_sobre_p0(M_sup, g)
    # salto de p do choque normal (Eq. 12), escrito aqui para evitar import circular
    # (bocal_choque.py importa este módulo; este módulo não pode importar bocal_choque.py)
    pII = pIII * (1.0 + 2.0 * g / (g + 1.0) * (M_sup**2 - 1.0))
    return pI, pII, pIII


# =============================================================================
## ESCOAMENTO TODO SUBSÔNICO (p_b >= pI, garganta NÃO sônica)
# =============================================================================

def perfil_subsonico_total(p_b, x, g=GAMMA, R=R_GAS):
    """
    Caso p_b >= p_I: sem estrangulamento, M<1 em todo o bocal e pe = p_b.
    A* efetiva (menor que A_t) vem de Ae/A* = f(Me) com Me dado por p_b.
    """
    x = np.asarray(x, dtype=float)
    A = A_perfil(x)
    Me = np.sqrt(2.0 / (g - 1.0) * ((P0 / p_b) ** ((g - 1.0) / g) - 1.0))
    A_star = A_SAIDA / razao_area(Me, g)
    M = np.array([mach_de_area(Ai / A_star, 'sub', g) for Ai in A])
    T = T0 * T_sobre_T0(M, g)
    p = P0 * p_sobre_p0(M, g)
    rho = p / (R * T)
    V = M * np.sqrt(g * R * T)
    return dict(x=x, A=A, M=M, p=p, T=T, rho=rho, V=V, p0=np.full_like(x, P0), mdot=rho * V * A)


# =============================================================================
## CLASSIFICAÇÃO AUTOMÁTICA DO REGIME
# =============================================================================

TOL_REL = 1e-9   # tolerância relativa p/ decidir "p_b == pressão crítica"

def classificar_regime(p_b):
    """Devolve (código, descrição) conforme a posição de p_b em relação a pI, pII, pIII."""
    pI, pII, pIII = pressoes_criticas()
    igual = lambda a, b: abs(a - b) <= TOL_REL * b      # compara com tolerância relativa
    if igual(p_b, pI):
        return "b", "primeira crítica (garganta sônica, divergente subsônico)"
    if p_b > pI:
        return "a", "todo subsônico (sem estrangulamento)"
    if igual(p_b, pII):
        return "d", "choque no plano de saída"
    if p_b > pII:
        return "c", "choque normal no divergente"
    if igual(p_b, pIII):
        return "f", "condição de projeto (bocal adaptado)"
    if p_b > pIII:
        return "e", "sobre-expandido"
    return "g", "subexpandido"


# =============================================================================
## BLOCO DE TESTE
# =============================================================================

if __name__ == "__main__":
    
    print("====================================================")
    print("      TESTE DO MÓDULO A: ISENTRÓPICO                ")
    print("====================================================\n")

    # 1. Geometria
    x, A = criar_malha(iniciocomprimento=0.0, totalcomprimento=0.1, numerodepontos=500)
    x_garganta, A_garganta, _ = localizar_garganta(x, A)

    # 2. Vazão mássica bloqueada
    m_dot_max = calcular_vazao_maxima(A_garganta, P0, T0, GAMMA, R_GAS)
    print("--- 1. VAZÃO MÁSSICA BLOQUEADA ---")
    print(f"Área da garganta (A*): {A_garganta:.8e} m²")
    print(f"Vazão mássica máx (m_dot): {m_dot_max:.6f} kg/s\n")

    # 3. Perfil isentrópico com garganta sônica e divergente subsônico (regime b: p_b = pI)
    M_sub, p_p0_sub, T_T0_sub, _, p_sub, _, u_sub = resolver_perfil_isentropico(
        x, A, A_garganta, x_garganta, subsonico_divergente=True, p0=P0, T0=T0, gamma=GAMMA, R=R_GAS
    )
    idx_garganta = np.argmin(np.abs(x - x_garganta))

    print("--- 2. GARGANTA SÔNICA, DIVERGENTE SUBSÔNICO (p_b = pI) ---")
    print(f"Entrada  (x = {x[0]:.2f}m) -> Mach: {M_sub[0]:.4f} | p/p0: {p_p0_sub[0]:.4f} | p: {p_sub[0]/1e5:.4f} bar | u: {u_sub[0]:.2f} m/s")
    print(f"Garganta (x = {x[idx_garganta]:.4f}m) -> Mach: {M_sub[idx_garganta]:.4f} | p/p0: {p_p0_sub[idx_garganta]:.4f} | p: {p_sub[idx_garganta]/1e5:.4f} bar | u: {u_sub[idx_garganta]:.2f} m/s")
    print(f"Saída    (x = {x[-1]:.2f}m) -> Mach: {M_sub[-1]:.4f} | p/p0: {p_p0_sub[-1]:.4f} | p: {p_sub[-1]/1e5:.4f} bar | u: {u_sub[-1]:.2f} m/s\n")

    # 4. Perfil supersônico no divergente (condição de projeto isentrópica)
    M_sup, p_p0_sup, T_T0_sup, _, p_sup, _, u_sup = resolver_perfil_isentropico(
        x, A, A_garganta, x_garganta, subsonico_divergente=False, p0=P0, T0=T0, gamma=GAMMA, R=R_GAS
    )

    print("--- 3. PERFIL SUPERSÔNICO NO DIVERGENTE (CONDIÇÃO DE PROJETO) ---")
    print(f"Garganta (x = {x[idx_garganta]:.4f}m) -> Mach: {M_sup[idx_garganta]:.4f} | p/p0: {p_p0_sup[idx_garganta]:.4f}")
    print(f"Saída    (x = {x[-1]:.2f}m) -> Mach: {M_sup[-1]:.4f} | p/p0: {p_p0_sup[-1]:.4f} | p: {p_sup[-1]/1e5:.4f} bar | u: {u_sup[-1]:.2f} m/s\n")

    # 5. Pressões críticas e regimes
    pI, pII, pIII = pressoes_criticas()
    print("--- 4. PRESSÕES CRÍTICAS ---")
    print(f"pI/p0 = {pI/P0:.4f} | pII/p0 = {pII/P0:.4f} | pIII/p0 = {pIII/P0:.4f}")
    for r in (0.95, 0.75, 0.40, 0.08):
        print(f"p_b/p0 = {r:.2f} -> regime {classificar_regime(r * P0)}")
