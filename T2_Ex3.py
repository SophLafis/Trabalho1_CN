from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================================
# Exercício 3 - PVI
# y' = y - t^2 + 1, y(0) = 1
#
# Métodos:
# - Runge-Kutta de 2ª ordem: método de Heun
# - Runge-Kutta de 4ª ordem: método clássico
# - Preditor-corretor de quatro passos: Adams-Bashforth-Moulton
# ==========================================================

T0 = 0.0
Y0 = 1.0
TF = 2.0


def f(t, y):
    """Função do PVI: y' = f(t,y)."""
    return y - t**2 + 1.0


def y_exata(t):
    """Solução exata do PVI."""
    return (t + 1.0)**2


def runge_kutta_2_heun(h, t0=T0, y0=Y0, tf=TF):
    """
    Método de Runge-Kutta de 2ª ordem na forma de Heun
    (Euler aperfeiçoado).
    """
    n = int(round((tf - t0) / h))
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + h*k1)
        y[i + 1] = y[i] + (h/2.0)*(k1 + k2)

    return t, y


def runge_kutta_4_classico(h, t0=T0, y0=Y0, tf=TF):
    """Método clássico de Runge-Kutta de 4ª ordem."""
    n = int(round((tf - t0) / h))
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2.0, y[i] + (h/2.0)*k1)
        k3 = f(t[i] + h/2.0, y[i] + (h/2.0)*k2)
        k4 = f(t[i] + h, y[i] + h*k3)

        y[i + 1] = y[i] + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

    return t, y


def preditor_corretor_4_passos(h, t0=T0, y0=Y0, tf=TF, n_correcoes=1):
    """
    Método preditor-corretor de quatro passos do tipo Adams-Bashforth-Moulton.
    Os valores iniciais y1, y2 e y3 são obtidos pelo RK4 clássico.
    """
    n = int(round((tf - t0) / h))
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros(n + 1)

    # Inicialização com RK4: y0, y1, y2, y3.
    _, y_rk4_ini = runge_kutta_4_classico(h, t0, y0, min(tf, t0 + 3*h))
    m = min(3, n)
    y[:m + 1] = y_rk4_ini[:m + 1]

    valores_f = np.zeros(n + 1)
    for j in range(m + 1):
        valores_f[j] = f(t[j], y[j])

    for i in range(3, n):
        # Adams-Bashforth de 4 passos: predição.
        y_p = y[i] + (h/24.0)*(
            55.0*valores_f[i]
            - 59.0*valores_f[i - 1]
            + 37.0*valores_f[i - 2]
            - 9.0*valores_f[i - 3]
        )

        # Adams-Moulton de 4 passos: correção.
        y_c = y[i] + (h/24.0)*(
            9.0*f(t[i + 1], y_p)
            + 19.0*valores_f[i]
            - 5.0*valores_f[i - 1]
            + valores_f[i - 2]
        )

        for _ in range(n_correcoes - 1):
            y_c = y[i] + (h/24.0)*(
                9.0*f(t[i + 1], y_c)
                + 19.0*valores_f[i]
                - 5.0*valores_f[i - 1]
                + valores_f[i - 2]
            )

        y[i + 1] = y_c
        valores_f[i + 1] = f(t[i + 1], y[i + 1])

    return t, y


def montar_tabela_amostra(h):
    """
    Monta uma tabela compacta para o texto principal.
    São mostrados valores em t = 0, 0.5, 1.0, 1.5 e 2.0.
    """
    t, y_rk2 = runge_kutta_2_heun(h)
    _, y_rk4 = runge_kutta_4_classico(h)

    pontos = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    indices = [int(round((p - T0)/h)) for p in pontos]

    t_amostra = t[indices]
    y_ref = y_exata(t_amostra)
    rk2 = y_rk2[indices]
    rk4 = y_rk4[indices]

    return pd.DataFrame({
        "t": t_amostra,
        "y_exata": y_ref,
        "RK2_Heun": rk2,
        "erro_RK2": np.abs(rk2 - y_ref),
        "RK4": rk4,
        "erro_RK4": np.abs(rk4 - y_ref),
    })


def montar_tabela_completa(h):
    """Gera a tabela completa em CSV, caso seja necessário consultar todos os pontos."""
    t, y_rk2 = runge_kutta_2_heun(h)
    _, y_rk4 = runge_kutta_4_classico(h)

    y_ref = y_exata(t)

    return pd.DataFrame({
        "t": t,
        "y_exata": y_ref,
        "RK2_Heun": y_rk2,
        "erro_RK2": np.abs(y_rk2 - y_ref),
        "RK4": y_rk4,
        "erro_RK4": np.abs(y_rk4 - y_ref),
    })


def montar_resumo(passos):
    """Monta tabela resumo em t = 2 para RK2, RK4 e preditor-corretor."""
    linhas = []

    for h in passos:
        _, y_rk2 = runge_kutta_2_heun(h)
        _, y_rk4 = runge_kutta_4_classico(h)
        _, y_pc4 = preditor_corretor_4_passos(h)

        y_ref = y_exata(TF)

        linhas.extend([
            {
                "Metodo": "RK2 (Heun)",
                "h": h,
                "y2_aproximado": y_rk2[-1],
                "erro_absoluto": abs(y_rk2[-1] - y_ref),
            },
            {
                "Metodo": "RK4 clássico",
                "h": h,
                "y2_aproximado": y_rk4[-1],
                "erro_absoluto": abs(y_rk4[-1] - y_ref),
            },
            {
                "Metodo": "Preditor-corretor",
                "h": h,
                "y2_aproximado": y_pc4[-1],
                "erro_absoluto": abs(y_pc4[-1] - y_ref),
            },
        ])

    return pd.DataFrame(linhas)


def montar_erros_rk(h):
    """Monta os erros absolutos de RK2 e RK4 ao longo do intervalo."""
    t, y_rk2 = runge_kutta_2_heun(h)
    _, y_rk4 = runge_kutta_4_classico(h)
    y_ref = y_exata(t)

    return pd.DataFrame({
        "t": t,
        "erro_RK2": np.abs(y_rk2 - y_ref),
        "erro_RK4": np.abs(y_rk4 - y_ref),
    })


def configurar_eixos(ax):
    """Aplica formatação comum aos gráficos."""
    ax.grid(True, which="both", color="0.82", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def gerar_grafico_erros_rk(tabela_erros, h, nome_arquivo):
    """
    Gera gráfico dos erros absolutos de RK2 e RK4 ao longo do intervalo.
    Este gráfico substitui a comparação direta das soluções, porque as curvas
    de y(t) ficam muito próximas e dificultam a visualização das diferenças.
    """
    cores = {
        "rk2": "#1f77b4",
        "rk4": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.9))

    t = tabela_erros["t"]

    ax.semilogy(
        t,
        tabela_erros["erro_RK2"].replace(0, np.nan),
        marker="o",
        markersize=5.5,
        markeredgecolor="black",
        linewidth=2.0,
        color=cores["rk2"],
        label="RK2 (Heun)"
    )

    ax.semilogy(
        t,
        tabela_erros["erro_RK4"].replace(0, np.nan),
        marker="s",
        markersize=5.2,
        markeredgecolor="black",
        linewidth=2.0,
        color=cores["rk4"],
        label="RK4 clássico"
    )

    ax.set_xlabel("$t$")
    ax.set_ylabel("Erro absoluto")
    ax.set_title(f"Erro absoluto dos métodos de Runge-Kutta para $h={str(h).replace('.', ',')}$")
    configurar_eixos(ax)
    ax.legend(frameon=True, framealpha=1.0)

    fig.tight_layout()
    fig.savefig(nome_arquivo, format="eps", bbox_inches="tight")
    plt.close(fig)


def gerar_grafico_erro_final(resumo, nome_arquivo):
    """Gera gráfico dos erros finais em t = 2."""
    cores = {
        "rk2": "#1f77b4",
        "rk4": "#d62728",
        "pc": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.9))

    for i, (_, linha) in enumerate(resumo.iterrows()):
        metodo = linha["Metodo"]

        if "RK2" in metodo:
            cor = cores["rk2"]
            marcador = "o"
        elif "RK4" in metodo:
            cor = cores["rk4"]
            marcador = "s"
        else:
            cor = cores["pc"]
            marcador = "^"

        ax.scatter(
            i,
            linha["erro_absoluto"],
            s=85,
            color=cor,
            marker=marcador,
            edgecolor="black",
            linewidth=0.8,
            zorder=3
        )

    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(resumo)))
    ax.set_xticklabels([
        f"{linha['Metodo']}\n$h={linha['h']:.1f}$"
        for _, linha in resumo.iterrows()
    ])

    ax.set_ylabel("Erro absoluto em $t=2$")
    ax.set_title("Comparação dos erros finais")
    configurar_eixos(ax)

    fig.tight_layout()
    fig.savefig(nome_arquivo, format="eps", bbox_inches="tight")
    plt.close(fig)


def gerar_graficos(erros_h02, erros_h01, resumo, pasta_imagens="Imagens"):
    """Gera todos os gráficos em EPS."""
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

    gerar_grafico_erros_rk(
        erros_h02,
        0.2,
        pasta_imagens / "ex3_erros_rk_h02.eps"
    )

    gerar_grafico_erros_rk(
        erros_h01,
        0.1,
        pasta_imagens / "ex3_erros_rk_h01.eps"
    )

    gerar_grafico_erro_final(
        resumo,
        pasta_imagens / "ex3_erro_final_t2.eps"
    )


def formatar_cientifico(valor):
    """Formata número para saída no terminal."""
    return f"{valor:.3e}"


def imprimir_resumo(resumo):
    """Imprime a tabela resumo no terminal."""
    print()
    print("="*76)
    print("RESUMO EM t = 2")
    print("="*76)
    print(f"{'Método':<20} | {'h':>4} | {'y(2) aproximado':>16} | {'Erro absoluto':>14}")
    print("-"*76)

    for _, linha in resumo.iterrows():
        print(
            f"{linha['Metodo']:<20} | "
            f"{linha['h']:>4.1f} | "
            f"{linha['y2_aproximado']:>16.10f} | "
            f"{formatar_cientifico(linha['erro_absoluto']):>14}"
        )


def main():
    Path("Dados").mkdir(exist_ok=True)
    Path("Imagens").mkdir(exist_ok=True)

    passos = [0.2, 0.1]

    # Tabelas completas e amostrais em CSV.
    for h in passos:
        sufixo = str(h).replace(".", "")
        montar_tabela_completa(h).to_csv(f"Dados/tabela_completa_h{sufixo}.csv", index=False)
        montar_tabela_amostra(h).to_csv(f"Dados/tabela_amostra_h{sufixo}.csv", index=False)

    erros_h02 = montar_erros_rk(0.2)
    erros_h01 = montar_erros_rk(0.1)
    resumo = montar_resumo(passos)

    erros_h02.to_csv("Dados/erros_rk_h02.csv", index=False)
    erros_h01.to_csv("Dados/erros_rk_h01.csv", index=False)
    resumo.to_csv("Dados/resumo_t2_ex3.csv", index=False)

    gerar_graficos(erros_h02, erros_h01, resumo, pasta_imagens="Imagens")
    imprimir_resumo(resumo)

    print()
    print("Arquivos gerados:")
    print("- Dados/tabela_completa_h02.csv")
    print("- Dados/tabela_completa_h01.csv")
    print("- Dados/tabela_amostra_h02.csv")
    print("- Dados/tabela_amostra_h01.csv")
    print("- Dados/erros_rk_h02.csv")
    print("- Dados/erros_rk_h01.csv")
    print("- Dados/resumo_t2_ex3.csv")
    print("- Imagens/ex3_erros_rk_h02.eps")
    print("- Imagens/ex3_erros_rk_h01.eps")
    print("- Imagens/ex3_erro_final_t2.eps")


if __name__ == "__main__":
    main()
