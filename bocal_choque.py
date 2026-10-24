"""
Módulo B: Choque normal, localização iterativa do choque, recomposição a jusante,
propriedades de estagnação e vazão bloqueada.


"""
import numpy as np
from scipy.optimize import brentq  # achador de raiz por bracketing

from parametros import GAMMA, R_GAS, P0, T0, L
# --- vem da Pessoa A ---
from bocal_isentropico import (A_perfil, dAdx_perfil, X_GARGANTA, A_GARGANTA, A_SAIDA,
                               razao_area, mach_de_area, p_sobre_p0, T_sobre_T0,
                               pressoes_criticas, perfil_subsonico_total)

# ============================================================================
# RELAÇÕES DO CHOQUE NORMAL 
# ============================================================================
def choque_normal(M1, g=GAMMA):
    """
    Choque normal. Entrada: M1 (> 1, Mach antes do choque).
    Saída: dict com M2, p2/p1, rho2/rho1, T2/T1, p02/p01 e delta_s/R.
    """
    if M1 < 1.0:
        # Segunda Lei: choque de expansão é impossível (delta_s < 0)
        raise ValueError("Choque normal exige M1 >= 1.")

    # Mach depois do choque 
    M2 = np.sqrt((M1**2 + 2.0 / (g - 1.0)) / (2.0 * g / (g - 1.0) * M1**2 - 1.0))
    # Salto de pressão(Eq. 12), vem do balanço de momento
    p2_p1 = 1.0 + 2.0 * g / (g + 1.0) * (M1**2 - 1.0)
    # Salto de densidade 
    rho2_rho1 = (g + 1.0) * M1**2 / (2.0 + (g - 1.0) * M1**2)
    # Salto de temperatura 
    T2_T1 = p2_p1 / rho2_rho1
    # Razão de pressões de estagnação (forma fechada)
    p02_p01 = (((g + 1.0) * M1**2 / (2.0 + (g - 1.0) * M1**2)) ** (g / (g - 1.0))
               * ((g + 1.0) / (2.0 * g * M1**2 - (g - 1.0))) ** (1.0 / (g - 1.0)))
    # Entropia gerada: T0 constante => delta_s/R = -ln(p02/p01)
    ds_R = -np.log(p02_p01)

    return dict(M2=M2, p2_p1=p2_p1, rho2_rho1=rho2_rho1, T2_T1=T2_T1,
                p02_p01=p02_p01, ds_R=ds_R)


# ============================================================================
# PROPRIEDADES DE ESTAGNAÇÃO ANTES E DEPOIS DO CHOQUE
# ============================================================================
def estagnacao_pos_choque(p01, T01, M1, R=R_GAS, g=GAMMA):
    """
    Dado o estado de estagnação antes do choque e M1, devolve:
    p02, T02, razão A*2/A*1 (= p01/p02, Eq. 20), perda de p0 e delta_s [J/(kg K)].
    """
    s = choque_normal(M1, g)
    p02 = p01 * s['p02_p01']            # p0 cai (choque gera entropia)
    T02 = T01                           # T0 se conserva (choque adiabático, h01 = h02)
    return dict(p02=p02, T02=T02,
                Astar2_Astar1=1.0 / s['p02_p01'],  # nova área sônica de referência (maior)
                perda_p0=p01 - p02,
                delta_s=R * s['ds_R'])             # J/(kg K)


# ============================================================================
# VAZÃO MÁSSICA BLOQUEADA (Eq. 4 / 58)
# ============================================================================
def vazao_bloqueada(p0=P0, T0_=T0, A_t=A_GARGANTA, R=R_GAS, g=GAMMA):
    """mdot_max = A* p0 sqrt(g/(R T0)) (2/(g+1))^((g+1)/(2(g-1)))."""
    return A_t * p0 * np.sqrt(g / (R * T0_)) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


# ============================================================================
# LOCALIZAÇÃO ITERATIVA DO CHOQUE
# ============================================================================
def estado_saida_com_choque(x_s, g=GAMMA):
    """
    Dada uma posição de choque x_s (no divergente), calcula a pressão na saída.
    Esta é a função "direta" cujo zero (pe - pb = 0) o solver procura.
    """
    A_s = A_perfil(x_s)
    # 1) Antes do choque: isentrópico, A* = A_t, ramo supersônico
    M1 = mach_de_area(A_s / A_GARGANTA, 'sup', g)
    # 2) Salto: novo p0 e nova área sônica A*2 = A*/(p02/p01)
    est = estagnacao_pos_choque(P0, T0, M1)
    A_star2 = A_GARGANTA * est['Astar2_Astar1']
    # 3) Depois do choque: subsônico até a saída com (p02, T0, A*2)
    Me = mach_de_area(A_SAIDA / A_star2, 'sub', g)
    pe = est['p02'] * p_sobre_p0(Me, g)
    return dict(M1=M1, p02=est['p02'], A_star2=A_star2, Me=Me, pe=pe)


def localizar_choque(p_b, tol=1e-12):
    """
    Acha x_s tal que pe(x_s) = p_b. Funciona se p_II < p_b < p_I (choque no divergente).
    pe(x_s) decresce com x_s: choque na garganta -> pe = p_I; choque na saída -> pe = p_II.
    Logo f(x_s) = pe - p_b troca de sinal nos extremos e brentq converge.
    """
    pI, pII, _ = pressoes_criticas()
    if not (pII < p_b < pI):
        raise ValueError(f"Sem choque no divergente: p_b = {p_b:.0f} Pa fora de ({pII:.0f}, {pI:.0f}) Pa.")
    f = lambda xs: estado_saida_com_choque(xs)['pe'] - p_b   # resíduo
    return brentq(f, X_GARGANTA, L, xtol=tol)


# ============================================================================
# RECOMPOSIÇÃO DO ESCOAMENTO AO LONGO DE TODO O BOCAL (com choque em x_s)
# ============================================================================
def perfil_com_choque(x, x_s, g=GAMMA, R=R_GAS):
    """
    Distribuição ao longo de x (array) para choque em x_s:
      x < x_t        : subsônico isentrópico (A*, p0)
      x_t <= x < x_s : supersônico isentrópico (A*, p0)
      x >= x_s       : subsônico isentrópico (A*2, p02)
    Devolve dict com M, p, T, rho, V, p0_local, mdot (arrays).
    """
    x = np.asarray(x, dtype=float)
    A = A_perfil(x)
    # Estado do choque
    M1 = mach_de_area(A_perfil(x_s) / A_GARGANTA, 'sup', g)
    est = estagnacao_pos_choque(P0, T0, M1)
    A_star2 = A_GARGANTA * est['Astar2_Astar1']

    M = np.empty_like(x)
    p0_loc = np.empty_like(x)
    for i, (xi, Ai) in enumerate(zip(x, A)):
        if xi < X_GARGANTA:                      # convergente: subsônico
            M[i] = mach_de_area(Ai / A_GARGANTA, 'sub', g); p0_loc[i] = P0
        elif xi < x_s:                           # divergente antes do choque: supersônico
            M[i] = mach_de_area(Ai / A_GARGANTA, 'sup', g); p0_loc[i] = P0
        else:                                    # depois do choque: subsônico, nova referência
            M[i] = mach_de_area(Ai / A_star2, 'sub', g);    p0_loc[i] = est['p02']

    T = T0 * T_sobre_T0(M, g)                    # T0 é o mesmo em todo o bocal
    p = p0_loc * p_sobre_p0(M, g)
    rho = p / (R * T)                            # equação de estado
    V = M * np.sqrt(g * R * T)                   # V = M a
    return dict(x=x, A=A, M=M, p=p, T=T, rho=rho, V=V, p0=p0_loc, mdot=rho * V * A)




# ============================================================================
# VALIDAÇÕES
# ============================================================================
def validar(res, p_b, mdot_ref=None, tem_choque=True):
    """Imprime: constância de mdot, p0 só muda no choque, pe = pb."""
    mdot = res['mdot']
    var = (mdot.max() - mdot.min()) / mdot.mean()
    print(f"  mdot médio = {mdot.mean():.5f} kg/s | variação relativa = {var:.2e}")
    if mdot_ref is not None:
        print(f"  mdot / mdot_max = {mdot.mean() / mdot_ref:.5f}")
    n_saltos = np.count_nonzero(np.abs(np.diff(res['p0'])) > 1e-6 * P0)
    print(f"  saltos de p0 ao longo de x = {n_saltos} (esperado: {1 if tem_choque else 0})")
    print(f"  p_saida = {res['p'][-1]:.1f} Pa | p_b = {p_b:.1f} Pa | erro = {abs(res['p'][-1] - p_b):.2e} Pa")


# ============================================================================
# TESTES
# ============================================================================
if __name__ == "__main__":
    # --- 1) Conferência contra exemplos resolvidos ---
    s = choque_normal(2.0)
    print("Choque M1=2  -> M2=%.4f (0.5774), p2/p1=%.3f (4.5), T2/T1=%.4f (1.6875), p02/p01=%.4f (0.7209)"
          % (s['M2'], s['p2_p1'], s['T2_T1'], s['p02_p01']))
    s = choque_normal(1.5)
    print("Choque M1=1.5 -> M2=%.3f (0.701), p02/p01=%.3f (0.930)" % (s['M2'], s['p02_p01']))
    s = choque_normal(3.0)
    print("Choque M1=3   -> delta_s = %.1f J/kgK (320)" % (287 * s['ds_R']))
    print("Vazão com A*=4e-4, p0=5e5, T0=300 -> %.4f kg/s (0.467)" % vazao_bloqueada(5e5, 300, 4e-4))

    print(f"\nGarganta: x_t = {X_GARGANTA:.5f} m, A* = {A_GARGANTA:.4e} m², dA/dx(x_t) = {dAdx_perfil(X_GARGANTA):.1e}")
    pI, pII, pIII = pressoes_criticas()
    print(f"Ae/A* = {A_SAIDA/A_GARGANTA:.3f} | pI/p0 = {pI/P0:.4f} | pII/p0 = {pII/P0:.4f} | pIII/p0 = {pIII/P0:.4f}")
    mdot_max = vazao_bloqueada()
    print(f"mdot_max = {mdot_max:.5f} kg/s\n")



