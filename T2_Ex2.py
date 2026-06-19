from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================
# Exercício 2 - Integração numérica
# Integral: I = integral_0^1 sqrt(1 + x^3) dx
# Métodos: Trapézio, Simpson 1/3, Simpson 3/8 e Gauss-Legendre
# =============================================================

I_REF = 1.1114479705
A = 0.0
B = 1.0
TOLERANCIA = 1e-5


def f(x):
    """Função integranda."""
    return np.sqrt(1.0 + x**3)


def trapezio_repetido(a, b, n):
    """Regra do trapézio composta."""
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n

    return (h / 2.0) * (y[0] + 2.0*np.sum(y[1:-1]) + y[-1])


def simpson_13_repetido(a, b, n):
    """Regra de Simpson 1/3 composta. Exige n par."""
    if n % 2 != 0:
        raise ValueError("Para Simpson 1/3 composto, n deve ser par.")

    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n

    soma_impares = np.sum(y[1:-1:2])
    soma_pares = np.sum(y[2:-1:2])

    return (h / 3.0) * (y[0] + 4.0*soma_impares + 2.0*soma_pares + y[-1])


def simpson_38_repetido(a, b, n):
    """Regra de Simpson 3/8 composta. Exige n múltiplo de 3."""
    if n % 3 != 0:
        raise ValueError("Para Simpson 3/8 composto, n deve ser múltiplo de 3.")

    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n

    soma = y[0] + y[-1]

    for i in range(1, n):
        if i % 3 == 0:
            soma += 2.0*y[i]
        else:
            soma += 3.0*y[i]

    return (3.0*h / 8.0) * soma


def gauss_legendre_2_pontos(a, b):
    """Quadratura de Gauss-Legendre com 2 pontos."""
    pontos = np.array([-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)])
    pesos = np.array([1.0, 1.0])

    # Mudança de variável de [-1, 1] para [a, b].
    x = (a + b)/2.0 + (b - a)*pontos/2.0

    return (b - a)/2.0 * np.sum(pesos*f(x))


def gauss_legendre_3_pontos(a, b):
    """Quadratura de Gauss-Legendre com 3 pontos."""
    pontos = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    pesos = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

    # Mudança de variável de [-1, 1] para [a, b].
    x = (a + b)/2.0 + (b - a)*pontos/2.0

    return (b - a)/2.0 * np.sum(pesos*f(x))


def f_segunda_derivada(x):
    """Segunda derivada de f(x) = sqrt(1 + x^3)."""
    return 3.0*x*(x**3 + 4.0) / (4.0*(1.0 + x**3)**(3.0/2.0))


def estimativa_n_trapezio(tolerancia=TOLERANCIA):
    """
    Estimativa analítica do número de subintervalos para a regra do trapézio.
    Usa |E_T| <= M2/(12*n^2), pois o intervalo é [0,1].
    """
    x_max = np.sqrt(3.0) - 1.0
    M2 = f_segunda_derivada(x_max)

    n_min = int(np.ceil(np.sqrt(M2 / (12.0*tolerancia))))

    return x_max, M2, n_min


def gerar_tabelas(n_valores):
    """Calcula as tabelas de resultados dos métodos numéricos."""
    linhas_nc = []

    for n in n_valores:
        I_trap = trapezio_repetido(A, B, n)
        I_s13 = simpson_13_repetido(A, B, n)
        I_s38 = simpson_38_repetido(A, B, n)

        linhas_nc.append({
            "n": n,
            "avaliacoes": n + 1,
            "Trapezio": I_trap,
            "Erro_Trapezio": abs(I_trap - I_REF),
            "Simpson_13": I_s13,
            "Erro_Simpson_13": abs(I_s13 - I_REF),
            "Simpson_38": I_s38,
            "Erro_Simpson_38": abs(I_s38 - I_REF),
        })

    tabela_nc = pd.DataFrame(linhas_nc)

    I_g2 = gauss_legendre_2_pontos(A, B)
    I_g3 = gauss_legendre_3_pontos(A, B)

    tabela_gauss = pd.DataFrame([
        {
            "Metodo": "Gauss-Legendre 2 pontos",
            "avaliacoes": 2,
            "Aproximacao": I_g2,
            "Erro": abs(I_g2 - I_REF),
        },
        {
            "Metodo": "Gauss-Legendre 3 pontos",
            "avaliacoes": 3,
            "Aproximacao": I_g3,
            "Erro": abs(I_g3 - I_REF),
        },
    ])

    return tabela_nc, tabela_gauss


def formatar_cientifico(valor):
    """Formata número em notação científica para impressão no terminal."""
    return f"{valor:.3e}"


def imprimir_tabelas(tabela_nc, tabela_gauss):
    """Imprime os resultados principais com arredondamento adequado."""
    print()
    print("="*78)
    print("TABELA - NEWTON-COTES")
    print("="*78)
    print(
        f"{'n':>5} | {'Trapézio':>12} | {'Erro':>10} | "
        f"{'Simpson 1/3':>12} | {'Erro':>10} | "
        f"{'Simpson 3/8':>12} | {'Erro':>10}"
    )
    print("-"*78)

    for _, linha in tabela_nc.iterrows():
        print(
            f"{int(linha['n']):5d} | "
            f"{linha['Trapezio']:12.8f} | {formatar_cientifico(linha['Erro_Trapezio']):>10} | "
            f"{linha['Simpson_13']:12.8f} | {formatar_cientifico(linha['Erro_Simpson_13']):>10} | "
            f"{linha['Simpson_38']:12.8f} | {formatar_cientifico(linha['Erro_Simpson_38']):>10}"
        )

    print()
    print("="*78)
    print("TABELA - QUADRATURA GAUSSIANA")
    print("="*78)
    print(f"{'Método':<28} | {'Avaliações':>10} | {'Aproximação':>12} | {'Erro':>10}")
    print("-"*78)

    for _, linha in tabela_gauss.iterrows():
        print(
            f"{linha['Metodo']:<28} | "
            f"{int(linha['avaliacoes']):10d} | "
            f"{linha['Aproximacao']:12.8f} | "
            f"{formatar_cientifico(linha['Erro']):>10}"
        )


def configurar_eixos(ax):
    """Aplica formatação comum aos gráficos."""
    ax.grid(True, which="both", color="0.78", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def gerar_graficos(tabela_nc, tabela_gauss, pasta_imagens="Imagens"):
    """Gera os gráficos em EPS para inclusão no LaTeX."""
    pasta_imagens = Path(pasta_imagens)
    pasta_imagens.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    cores = {
        "trapezio": "#d62728",
        "simpson_13": "#1f77b4",
        "simpson_38": "#2ca02c",
        "gauss": "#9467bd",
        "tolerancia": "0.25",
    }

    # ---------------------------------------------------------
    # Gráfico 1: erro absoluto em função do número de subintervalos.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    ax.semilogy(
        tabela_nc["n"],
        tabela_nc["Erro_Trapezio"],
        marker="o",
        markersize=6,
        markeredgecolor="black",
        linewidth=2.0,
        color=cores["trapezio"],
        label="Trapézio"
    )

    ax.semilogy(
        tabela_nc["n"],
        tabela_nc["Erro_Simpson_13"],
        marker="s",
        markersize=6,
        markeredgecolor="black",
        linewidth=2.0,
        color=cores["simpson_13"],
        label="Simpson 1/3"
    )

    ax.semilogy(
        tabela_nc["n"],
        tabela_nc["Erro_Simpson_38"],
        marker="^",
        markersize=7,
        markeredgecolor="black",
        linewidth=2.0,
        color=cores["simpson_38"],
        label="Simpson 3/8"
    )

    ax.axhline(
        TOLERANCIA,
        color=cores["tolerancia"],
        linestyle="--",
        linewidth=1.4,
        label=r"Tolerância $10^{-5}$"
    )

    ax.set_xlabel("Número de subintervalos (n)")
    ax.set_ylabel("Erro absoluto")
    ax.set_title("Erro absoluto em função do número de subintervalos")
    ax.set_xticks(tabela_nc["n"])
    configurar_eixos(ax)
    ax.legend(frameon=True, framealpha=1.0)

    fig.tight_layout()
    fig.savefig(
        pasta_imagens / "erro_vs_n.eps",
        format="eps",
        bbox_inches="tight"
    )
    plt.close(fig)

    # ---------------------------------------------------------
    # Gráfico 2: erro absoluto em função do número de avaliações de f.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    ax.loglog(
        tabela_nc["avaliacoes"],
        tabela_nc["Erro_Trapezio"],
        marker="o",
        markersize=6,
        markeredgecolor="black",
        linewidth=2.0,
        color=cores["trapezio"],
        label="Trapézio"
    )

    ax.loglog(
        tabela_nc["avaliacoes"],
        tabela_nc["Erro_Simpson_13"],
        marker="s",
        markersize=6,
        markeredgecolor="black",
        linewidth=2.0,
        color=cores["simpson_13"],
        label="Simpson 1/3"
    )

    ax.loglog(
        tabela_nc["avaliacoes"],
        tabela_nc["Erro_Simpson_38"],
        marker="^",
        markersize=7,
        markeredgecolor="black",
        linewidth=2.0,
        color=cores["simpson_38"],
        label="Simpson 3/8"
    )

    ax.loglog(
        tabela_gauss["avaliacoes"],
        tabela_gauss["Erro"],
        marker="D",
        markersize=7,
        markeredgecolor="black",
        linestyle="--",
        linewidth=1.8,
        color=cores["gauss"],
        label="Gauss-Legendre"
    )

    ax.set_xlabel("Número de avaliações da função")
    ax.set_ylabel("Erro absoluto")
    ax.set_title("Erro absoluto em função das avaliações de f")
    configurar_eixos(ax)
    ax.legend(frameon=True, framealpha=1.0)

    fig.tight_layout()
    fig.savefig(
        pasta_imagens / "erro_vs_avaliacoes.eps",
        format="eps",
        bbox_inches="tight"
    )
    plt.close(fig)


def main():
    n_valores = [6, 12, 24, 48, 96]

    tabela_nc, tabela_gauss = gerar_tabelas(n_valores)
    x_max, M2, n_min = estimativa_n_trapezio(TOLERANCIA)

    Path("Dados").mkdir(exist_ok=True)
    tabela_nc.to_csv("Dados/resultados_newton_cotes.csv", index=False)
    tabela_gauss.to_csv("Dados/resultados_gauss.csv", index=False)

    gerar_graficos(tabela_nc, tabela_gauss, pasta_imagens="Imagens")
    imprimir_tabelas(tabela_nc, tabela_gauss)

    valor_trap_n_min = trapezio_repetido(A, B, n_min)
    erro_trap_n_min = abs(valor_trap_n_min - I_REF)

    print()
    print("="*78)
    print("ESTIMATIVA ANALÍTICA PARA A REGRA DO TRAPÉZIO")
    print("="*78)
    print(f"x onde |f''(x)| é máximo: {x_max:.10f}")
    print(f"M2 = max |f''(x)|:        {M2:.10f}")
    print(f"n mínimo estimado:        {n_min}")
    print(f"Trapézio com n = {n_min}:     {valor_trap_n_min:.10f}")
    print(f"Erro absoluto:            {erro_trap_n_min:.10e}")

    print()
    print("Arquivos gerados:")
    print("- Imagens/erro_vs_n.eps")
    print("- Imagens/erro_vs_avaliacoes.eps")
    print("- Dados/resultados_newton_cotes.csv")
    print("- Dados/resultados_gauss.csv")


if __name__ == "__main__":
    main()
