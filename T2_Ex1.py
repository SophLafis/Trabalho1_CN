import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# Funções auxiliares para a saída do terminal
# ---------------------------------------------------------

def linha(tamanho=72):
    print("-" * tamanho)


def titulo(texto):
    print()
    linha()
    print(texto.center(72))
    linha()


def imprimir_sistema_normal(A, b):
    titulo("SISTEMA NORMAL DOS MÍNIMOS QUADRADOS")

    print("Sistema na forma:")
    print()
    print("    [ A11  A12 ] [ c0 ] = [ b1 ]")
    print("    [ A21  A22 ] [ c1 ]   [ b2 ]")
    print()

    print("Com os valores calculados:")
    print()
    print(f"    [ {A[0,0]:10.6f}  {A[0,1]:10.6f} ] [ c0 ] = [ {b[0]:12.6f} ]")
    print(f"    [ {A[1,0]:10.6f}  {A[1,1]:10.6f} ] [ c1 ]   [ {b[1]:12.6f} ]")


def imprimir_tabela(t, N, z, N_ajustado):
    titulo("TABELA DE COMPARAÇÃO")

    print(
        f"{'t':>5} | {'N observado':>14} | {'ln(N)':>12} | "
        f"{'N ajustado':>14} | {'resíduo':>12} | {'resíduo rel. (%)':>16}"
    )
    linha(92)

    for ti, Ni, zi, Nai in zip(t, N, z, N_ajustado):
        residuo = Ni - Nai
        residuo_relativo = 100*residuo/Ni
        print(
            f"{ti:5.0f} | {Ni:14.2f} | {zi:12.6f} | "
            f"{Nai:14.2f} | {residuo:12.2f} | {residuo_relativo:16.2f}"
        )

    linha(92)


def configurar_grafico(ax):
    ax.grid(True, color="0.78", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def salvar_graficos(t, N, z, c0, c1, modelo, R2, pasta_imagens):
    pasta_imagens.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # -----------------------------------------------------
    # Gráfico 1: dados observados e ajuste exponencial
    # no intervalo dos dados.
    # -----------------------------------------------------

    t_modelo = np.linspace(t.min(), t.max(), 300)
    N_modelo = modelo(t_modelo)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    ax.scatter(
        t,
        N,
        s=52,
        color="#d62728",
        edgecolor="black",
        linewidth=0.6,
        label="Dados observados",
        zorder=3
    )

    ax.plot(
        t_modelo,
        N_modelo,
        color="#1f77b4",
        linewidth=2.1,
        label="Modelo exponencial ajustado"
    )

    ax.set_xlabel("t (horas)")
    ax.set_ylabel("N(t)")
    ax.set_title("Diagrama de dispersão e ajuste exponencial")
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(0, max(N)*1.12)
    configurar_grafico(ax)
    ax.legend(frameon=True, framealpha=1.0)

    fig.tight_layout()
    fig.savefig(
        pasta_imagens / "ex1_ajuste_exponencial.eps",
        format="eps",
        bbox_inches="tight"
    )
    plt.close(fig)

    # -----------------------------------------------------
    # Gráfico 2: teste de alinhamento do modelo exponencial.
    # -----------------------------------------------------

    t_alinhamento = np.linspace(t.min(), t.max(), 300)
    z_ajustado_grafico = c0 + c1*t_alinhamento

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    ax.scatter(
        t,
        z,
        s=52,
        color="#9467bd",
        edgecolor="black",
        linewidth=0.6,
        label="Dados linearizados",
        zorder=3
    )

    ax.plot(
        t_alinhamento,
        z_ajustado_grafico,
        color="#2ca02c",
        linewidth=2.1,
        label=f"Reta ajustada ($R^2={R2:.6f}$)"
    )

    ax.set_xlabel("t (horas)")
    ax.set_ylabel("ln(N(t))")
    ax.set_title("Teste de alinhamento")
    ax.set_xlim(-0.4, 10.4)
    configurar_grafico(ax)
    ax.legend(frameon=True, framealpha=1.0)

    fig.tight_layout()
    fig.savefig(
        pasta_imagens / "ex1_teste_alinhamento.eps",
        format="eps",
        bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------
# Dados do exercício
# ---------------------------------------------------------

t = np.array([0, 1, 2, 3, 5, 7, 10], dtype=float)
N = np.array([100, 168, 266, 462, 1182, 3444, 14248], dtype=float)

# ---------------------------------------------------------
# Modelo escolhido:
#
# N(t) = alpha * exp(beta*t)
#
# Linearização:
# ln(N) = c0 + c1*t,
# em que c0 = ln(alpha) e c1 = beta.
# ---------------------------------------------------------

z = np.log(N)

# ---------------------------------------------------------
# Montagem do sistema normal dos mínimos quadrados
# ---------------------------------------------------------

m = len(t)

A = np.array([
    [m, np.sum(t)],
    [np.sum(t), np.sum(t**2)]
])

b = np.array([
    np.sum(z),
    np.sum(t*z)
])

# Resolução do sistema normal
c0, c1 = np.linalg.solve(A, b)

# Retorno para os parâmetros do modelo exponencial
alpha = np.exp(c0)
beta = c1


def modelo(x):
    return alpha * np.exp(beta*x)


# Valores ajustados nos pontos observados
N_ajustado = modelo(t)
z_ajustado = c0 + c1*t

# Coeficiente de correlação e R² no problema linearizado
r = np.corrcoef(t, z)[0, 1]
SQ_res = np.sum((z - z_ajustado)**2)
SQ_tot = np.sum((z - np.mean(z))**2)
R2 = 1 - SQ_res/SQ_tot

# Estimativas pedidas no enunciado
N_15 = modelo(15)
t_1000 = (np.log(1000) - c0)/c1

# ---------------------------------------------------------
# Impressão dos resultados
# ---------------------------------------------------------

imprimir_sistema_normal(A, b)

titulo("COEFICIENTES DO MODELO LINEARIZADO")
print(f"c0 = {c0:.6f}")
print(f"c1 = {c1:.6f}")

titulo("MODELO EXPONENCIAL AJUSTADO")
print(f"N(t) = {alpha:.6f} * exp({beta:.6f} t)")

titulo("TESTE DE ALINHAMENTO")
print(f"r  = {r:.6f}")
print(f"R² = {R2:.6f}")

titulo("ESTIMATIVAS")
print(f"N(15) = {N_15:.2f}")
print(f"Instante em que N(t) = 1000: t = {t_1000:.4f} horas")

imprimir_tabela(t, N, z, N_ajustado)

# ---------------------------------------------------------
# Geração dos gráficos em EPS
# ---------------------------------------------------------

PASTA_IMAGENS = Path("Imagens")
salvar_graficos(t, N, z, c0, c1, modelo, R2, PASTA_IMAGENS)

print()
print("Gráficos salvos em:")
print(PASTA_IMAGENS / "ex1_ajuste_exponencial.eps")
print(PASTA_IMAGENS / "ex1_teste_alinhamento.eps")
