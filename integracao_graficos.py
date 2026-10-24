"""
Integração e gráficos - usa o módulo de choque (bocal_choque.py).
Faz: (1) classificação automática do regime, (2) resolução de cada caso de teste,
(3) validações, (4) gráficos de A(x), M(x), p/p0(x) e T/T0(x) por caso e comparativo.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # backend sem janela (salva direto em arquivo)
import matplotlib.pyplot as plt

import parametros as P               # constantes compartilhadas
import bocal_isentropico as A        # módulo A
import bocal_choque as B             # módulo B

# ----------------------------------------------------------------------------
# Resolve um caso:
# ----------------------------------------------------------------------------
def resolver_caso(p_b, x):
    """Devolve (resultado, x_s, código, descrição). x_s = None se não houver choque interno."""
    cod, desc = A.classificar_regime(p_b)
    if cod in ("a", "b"):                     # subsônico (em 'b' a garganta é exatamente M=1)
        res, x_s = A.perfil_subsonico_total(p_b, x), None
    elif cod == "c":                          # choque dentro do divergente: solver iterativo
        x_s = B.localizar_choque(p_b)
        res = B.perfil_com_choque(x, x_s)
    elif cod == "d":                          # choque exatamente na saída
        x_s = P.L
        res = B.perfil_com_choque(x, x_s)
    else:                                     # e, f, g: interior = escoamento de projeto
        x_s = None
        res = B.perfil_com_choque(x, P.L + 1.0)   # x_s fora do bocal => sem choque
    return res, x_s, cod, desc


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def plotar_caso(res, x_s, titulo, arquivo, p_b):
    """4 painéis: A(x), M(x), p/p0(x), T/T0(x). Linha vertical tracejada = choque."""
    x = res["x"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    series = [
        (ax[0, 0], res["A"] * 1e4,        "A(x) [cm²]"),
        (ax[0, 1], res["M"],              "M(x)"),
        (ax[1, 0], res["p"] / P.P0,       "p/p0(x)"),
        (ax[1, 1], res["T"] / P.T0,       "T/T0(x)"),
    ]
    for a, y, rotulo in series:
        a.plot(x, y, lw=2)
        a.axvline(A.X_GARGANTA, color="gray", ls=":", lw=1, label="garganta")
        if x_s is not None:
            a.axvline(x_s, color="red", ls="--", lw=1, label="choque")
        a.set_ylabel(rotulo)
        a.grid(alpha=0.3)
    ax[0, 1].axhline(1.0, color="k", lw=0.8, ls="-.")           # linha M = 1
    ax[1, 0].axhline(p_b / P.P0, color="green", ls="--", lw=1, label="p_b/p0")  # contra-pressão
    ax[1, 0].legend(fontsize=8); ax[0, 0].legend(fontsize=8)
    for a in ax[1]:
        a.set_xlabel("x [m]")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def plotar_comparativo(resultados, arquivo):
    """Sobrepõe todos os casos em M(x), p/p0(x) e T/T0(x)."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for rotulo, res, x_s in resultados:
        l, = ax[0].plot(res["x"], res["M"], label=rotulo)
        ax[1].plot(res["x"], res["p"] / P.P0, color=l.get_color())
        ax[2].plot(res["x"], res["T"] / P.T0, color=l.get_color())
        if x_s is not None and x_s < P.L:                        # marca onde está o choque
            ax[0].axvline(x_s, color=l.get_color(), ls=":", lw=0.8)
    for a, y in zip(ax, ("M(x)", "p/p0(x)", "T/T0(x)")):
        a.set_xlabel("x [m]"); a.set_ylabel(y); a.grid(alpha=0.3)
    ax[0].axhline(1.0, color="k", lw=0.8, ls="-.")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Programa principal:
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    x = np.linspace(0.0, P.L, 2001)
    pI, pII, pIII = A.pressoes_criticas()
    mdot_max = B.vazao_bloqueada()
    print(f"pI/p0={pI/P.P0:.4f}  pII/p0={pII/P.P0:.4f}  pIII/p0={pIII/P.P0:.4f}  mdot_max={mdot_max:.5f} kg/s\n")

    # casos: razões p_b/p0 do e-mail + condição de projeto (p_III calculada pela rotina)
    casos = [("p_b/p0 = 0,95", 0.95 * P.P0), ("p_b/p0 = 0,75", 0.75 * P.P0),
             ("p_b/p0 = 0,40", 0.40 * P.P0), ("p_b/p0 = 0,08", 0.08 * P.P0),
             (f"projeto (p_b/p0 = {pIII/P.P0:.4f})", pIII)]

    resultados = []
    for i, (nome, p_b) in enumerate(casos, start=1):
        res, x_s, cod, desc = resolver_caso(p_b, x)
        print(f"=== {nome} -> ({cod}) {desc}")
        if x_s is not None and x_s < P.L:
            est = B.estado_saida_com_choque(x_s)
            e2 = B.estagnacao_pos_choque(P.P0, P.T0, est["M1"])
            print(f"  choque em x_s = {x_s:.5f} m (A_s/A* = {A.A_perfil(x_s)/A.A_GARGANTA:.4f})")
            print(f"  M1 = {est['M1']:.4f} -> M2 = {B.choque_normal(est['M1'])['M2']:.4f}")
            print(f"  p02/p01 = {e2['p02']/P.P0:.4f} | perda de p0 = {e2['perda_p0']/1e3:.2f} kPa | "
                  f"delta_s = {e2['delta_s']:.2f} J/(kg K) | T02 = {e2['T02']:.1f} K | A*2/A* = {e2['Astar2_Astar1']:.4f}")
        tem_choque = (x_s is not None)
        B.validar(res, p_b, mdot_max, tem_choque=tem_choque)
        if cod in ("e", "g"):
            print(f"  (pe interno = {res['p'][-1]:.0f} Pa difere de p_b = {p_b:.0f} Pa: ajuste ocorre no jato)")
        plotar_caso(res, x_s, f"{nome} - regime ({cod}): {desc}", f"caso{i}.png", p_b)
        resultados.append((nome, res, x_s))
        print()

    plotar_comparativo(resultados, "comparativo.png")
    print("Figuras salvas: caso1.png ... caso5.png e comparativo.png")
