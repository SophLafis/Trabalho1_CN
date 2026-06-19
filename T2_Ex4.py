from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# Exercício 4 - PVC por diferenças finitas centradas
#
# Problema:
#     y''(x) - y(x) = -x,   y(0)=0, y(1)=0
#
# Solução exata:
#     y(x) = x - sinh(x)/sinh(1)
#
# Método de solução do sistema:
#     Método de Thomas, adequado para matrizes tridiagonais.
# ============================================================

PASTA_IMAGENS = Path("Imagens")
PASTA_DADOS = Path("Dados")


def y_exata(x):
    """Solução exata do PVC."""
    x = np.asarray(x, dtype=float)
    return x - np.sinh(x)/np.sinh(1.0)


def montar_sistema(n):
    """
    Monta o sistema Ay = b usando diferenças finitas centradas.

    A discretização é feita nos pontos internos

        x_i = i h,    i = 1, 2, ..., n-1,

    com h = 1/n.

    Da equação

        y''(x_i) - y_i = -x_i,

    usando

        y''(x_i) ≈ (y_{i-1} - 2y_i + y_{i+1})/h²,

    obtém-se

        y_{i-1} - (2+h²)y_i + y_{i+1} = -h² x_i.

    Multiplicando por -1, usa-se a forma

        -y_{i-1} + (2+h²)y_i - y_{i+1} = h² x_i.
    """
    if n < 2:
        raise ValueError("n deve ser pelo menos 2.")

    h = 1.0/n
    m = n - 1
    x_int = np.linspace(h, 1.0 - h, m)

    A = np.zeros((m, m), dtype=float)
    np.fill_diagonal(A, 2.0 + h**2)

    for i in range(m - 1):
        A[i, i + 1] = -1.0
        A[i + 1, i] = -1.0

    b = h**2 * x_int

    # Como y(0)=0 e y(1)=0, não há contribuição adicional das fronteiras no vetor b.
    # Para condições de contorno não nulas, esses termos seriam somados em b[0] e b[-1].
    return A, b, x_int, h


def resolver_thomas(a, d, c, b):
    """
    Resolve um sistema tridiagonal pelo método de Thomas.

    Parâmetros:
        a: subdiagonal, tamanho m-1
        d: diagonal principal, tamanho m
        c: superdiagonal, tamanho m-1
        b: vetor independente, tamanho m
    """
    a = np.array(a, dtype=float).copy()
    d = np.array(d, dtype=float).copy()
    c = np.array(c, dtype=float).copy()
    b = np.array(b, dtype=float).copy()

    m = len(d)

    # Eliminação direta.
    for i in range(1, m):
        fator = a[i - 1] / d[i - 1]
        d[i] = d[i] - fator*c[i - 1]
        b[i] = b[i] - fator*b[i - 1]

    # Substituição regressiva.
    y = np.zeros(m, dtype=float)
    y[-1] = b[-1] / d[-1]

    for i in range(m - 2, -1, -1):
        y[i] = (b[i] - c[i]*y[i + 1]) / d[i]

    return y


def resolver_pvc(n):
    """Resolve o PVC para um dado número de subintervalos n."""
    A, b, x_int, h = montar_sistema(n)
    m = n - 1

    diagonal = np.full(m, 2.0 + h**2)
    subdiagonal = np.full(m - 1, -1.0)
    superdiagonal = np.full(m - 1, -1.0)

    y_int = resolver_thomas(subdiagonal, diagonal, superdiagonal, b)

    x = np.linspace(0.0, 1.0, n + 1)
    y = np.zeros(n + 1)
    y[1:-1] = y_int

    y_ex = y_exata(x)
    erro = np.abs(y - y_ex)

    residuo = A @ y_int - b
    norma_residuo = np.linalg.norm(residuo, ord=2)

    return x, y, y_ex, erro, A, b, h, norma_residuo


def tabela_resultados_n4():
    """Gera tabela de resultados para n=4, apenas nos pontos internos."""
    x, y, y_ex, erro, A, b, h, norma_residuo = resolver_pvc(4)

    tabela = pd.DataFrame({
        "x": x[1:-1],
        "y_aproximado": y[1:-1],
        "y_exato": y_ex[1:-1],
        "erro_abs": erro[1:-1],
    })

    return tabela, A, b, h, norma_residuo


def tabela_convergencia(valores_n):
    """Gera tabela de convergência do erro máximo."""
    linhas = []
    erro_anterior = None

    for n in valores_n:
        x, y, y_ex, erro, A, b, h, norma_residuo = resolver_pvc(n)
        erro_max = np.max(erro)

        if erro_anterior is None:
            ordem = np.nan
        else:
            ordem = np.log(erro_anterior/erro_max)/np.log(2.0)

        linhas.append({
            "n": n,
            "h": h,
            "erro_maximo": erro_max,
            "ordem_observada": ordem,
            "residuo_sistema": norma_residuo,
        })

        erro_anterior = erro_max

    return pd.DataFrame(linhas)


def configurar_eixos(ax):
    """Aplica formatação comum aos gráficos."""
    ax.grid(True, which="both", color="0.82", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def gerar_graficos(tabela_n4, convergencia, pasta_imagens=PASTA_IMAGENS):
    """Gera os gráficos em EPS para o relatório."""
    pasta_imagens.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # --------------------------------------------------------
    # Gráfico 1: erro absoluto nos pontos internos para n=4.
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.8, 4.9))

    ax.plot(
        tabela_n4["x"],
        tabela_n4["erro_abs"],
        marker="o",
        markersize=7,
        markeredgecolor="black",
        linewidth=2.0,
        color="#d62728",
        label="Erro absoluto"
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("Erro absoluto")
    ax.set_title("Erro absoluto nos pontos internos para $n=4$")
    ax.set_xticks(tabela_n4["x"])
    configurar_eixos(ax)
    ax.legend(frameon=True, framealpha=1.0)

    fig.tight_layout()
    fig.savefig(
        pasta_imagens / "ex4_erro_n4.eps",
        format="eps",
        bbox_inches="tight"
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Gráfico 2: convergência do erro máximo.
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.8, 4.9))

    h_vals = convergencia["h"].to_numpy()
    erro_vals = convergencia["erro_maximo"].to_numpy()

    ax.loglog(
        h_vals,
        erro_vals,
        marker="o",
        markersize=6,
        markeredgecolor="black",
        linewidth=2.0,
        color="#1f77b4",
        label="Erro máximo"
    )

    # Linha de referência proporcional a h².
    ref_h2 = erro_vals[0]*(h_vals/h_vals[0])**2

    ax.loglog(
        h_vals,
        ref_h2,
        linestyle="--",
        linewidth=1.8,
        color="#2ca02c",
        label="Referência proporcional a $h^2$"
    )

    ax.set_xlabel("Passo $h$")
    ax.set_ylabel("Erro absoluto máximo")
    ax.set_title("Convergência do método de diferenças finitas centradas")
    configurar_eixos(ax)
    ax.legend(frameon=True, framealpha=1.0)

    fig.tight_layout()
    fig.savefig(
        pasta_imagens / "ex4_convergencia_erro.eps",
        format="eps",
        bbox_inches="tight"
    )
    plt.close(fig)


def formatar_cientifico(valor):
    """Formata número para saída no terminal."""
    return f"{valor:.3e}"


def main():
    PASTA_DADOS.mkdir(exist_ok=True)
    PASTA_IMAGENS.mkdir(exist_ok=True)

    tabela_n4, A4, b4, h4, residuo_n4 = tabela_resultados_n4()
    convergencia = tabela_convergencia([4, 8, 16, 32, 64, 128])

    tabela_n4.to_csv(PASTA_DADOS / "resultados_n4.csv", index=False)
    convergencia.to_csv(PASTA_DADOS / "convergencia_ex4.csv", index=False)

    gerar_graficos(tabela_n4, convergencia, PASTA_IMAGENS)

    print("="*72)
    print("SISTEMA PARA n = 4")
    print("="*72)
    print(f"h = {h4:.6f}")
    print("A =")
    print(A4)
    print("b =")
    print(b4)
    print(f"||A y - b||_2 = {residuo_n4:.6e}")

    print()
    print("="*72)
    print("RESULTADOS PARA n = 4")
    print("="*72)
    print(tabela_n4.to_string(index=False))

    print()
    print("="*72)
    print("TABELA DE CONVERGÊNCIA")
    print("="*72)
    print(convergencia.to_string(index=False))

    print()
    print("Arquivos gerados:")
    print("- Dados/resultados_n4.csv")
    print("- Dados/convergencia_ex4.csv")
    print("- Imagens/ex4_erro_n4.eps")
    print("- Imagens/ex4_convergencia_erro.eps")


if __name__ == "__main__":
    main()
