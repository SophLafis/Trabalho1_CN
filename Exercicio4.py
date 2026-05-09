# Dicente Sophia Lafis Formiga, GRR 20246031
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# EXERCÍCIO 4 - INTERPOLAÇÃO DO MOVIMENTO DE UM PROJÉTIL
# Usar diferenças divididas de Newton para estimar y(5.5)
# ============================================================

# ============================================================
# DADOS DO PROBLEMA
# ============================================================

t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

y = np.array([0.00, 1.59, 2.85, 3.78, 4.46, 4.93,
              5.23, 5.38, 5.43, 5.37, 5.23], dtype=float)

valor_alvo = 5.5


# ============================================================
# FUNÇÕES AUXILIARES DE IMPRESSÃO
# ============================================================

def imprimir_titulo(titulo):
    print("\n" + "=" * 90)
    print(titulo)
    print("=" * 90)


def imprimir_vetor_dados(t, y):
    print("\nDados fornecidos no enunciado")
    print("-" * 40)
    print(f"{'t (s)':>10} | {'y(t) (m)':>15}")
    print("-" * 40)

    for ti, yi in zip(t, y):
        print(f"{ti:>10.2f} | {yi:>15.5f}")

    print("-" * 40)


# ============================================================
# TABELA DE DIFERENÇAS DIVIDIDAS
# ============================================================

def diferencas_divididas(x, y):
    """
    Constrói a tabela de diferenças divididas de Newton.
    """
    n = len(x)
    tabela = np.zeros((n, n), dtype=float)

    # Ordem zero
    tabela[:, 0] = y

    # Ordens superiores
    for j in range(1, n):
        for i in range(n - j):
            tabela[i, j] = (tabela[i + 1, j - 1] - tabela[i, j - 1]) / (x[i + j] - x[i])

    return tabela


def imprimir_tabela_diferencas(x, tabela):
    """
    Imprime a tabela de diferenças divididas de forma organizada.
    """
    n = len(x)

    print("\nTabela de diferenças divididas")
    print("-" * (14 + 16 * n))

    cabecalho = f"{'t':>10} |"
    for j in range(n):
        cabecalho += f" {'Ordem ' + str(j):>13} |"
    print(cabecalho)

    print("-" * (14 + 16 * n))

    for i in range(n):
        linha = f"{x[i]:>10.2f} |"

        for j in range(n):
            if j <= n - i - 1:
                linha += f" {tabela[i, j]:>13.6f} |"
            else:
                linha += f" {'':>13} |"

        print(linha)

    print("-" * (14 + 16 * n))


# ============================================================
# POLINÔMIO DE NEWTON
# ============================================================

def avaliar_newton(x_pontos, tabela, valor, grau):
    """
    Avalia o polinômio interpolador de Newton de um certo grau.
    """
    resultado = tabela[0, 0]
    produto = 1.0

    for j in range(1, grau + 1):
        produto *= (valor - x_pontos[j - 1])
        resultado += tabela[0, j] * produto

    return resultado

# ============================================================
# GRÁFICO
# ============================================================

def gerar_grafico(t, y, x_local, coeficientes, valor_alvo, valor_estimado):
    """
    Gera um gráfico dos dados do projétil, do polinômio local de grau 2
    e do polinômio local de grau 3 usado apenas para comparação.
    """
    # ----------------------------
    # Polinômio de grau 2
    # ----------------------------
    a0, a1, a2 = coeficientes
    x0, x1, x2 = x_local

    t_plot_grau2 = np.linspace(x0, x2, 300)

    p2_plot = (
        a0
        + a1 * (t_plot_grau2 - x0)
        + a2 * (t_plot_grau2 - x0) * (t_plot_grau2 - x1)
    )

    # ----------------------------
    # Polinômio de grau 3
    # Pontos usados: 5, 6, 7 e 8
    # ----------------------------
    x_local_grau3 = np.array([5, 6, 7, 8], dtype=float)
    y_local_grau3 = np.array([4.93, 5.23, 5.38, 5.43], dtype=float)

    tabela_grau3 = diferencas_divididas(x_local_grau3, y_local_grau3)

    t_plot_grau3 = np.linspace(x_local_grau3[0], x_local_grau3[-1], 300)

    p3_plot = np.array([
        avaliar_newton(
            x_pontos=x_local_grau3,
            tabela=tabela_grau3,
            valor=ti,
            grau=3
        )
        for ti in t_plot_grau3
    ])

    valor_estimado_grau3 = avaliar_newton(
        x_pontos=x_local_grau3,
        tabela=tabela_grau3,
        valor=valor_alvo,
        grau=3
    )

    diferenca = abs(valor_estimado_grau3 - valor_estimado)

    # ----------------------------
    # Construção do gráfico
    # ----------------------------
    plt.figure(figsize=(9, 5.5))

    plt.plot(t, y, "o", label="Dados observados")
    plt.plot(t_plot_grau2, p2_plot, "-", label="Polinômio interpolador de grau 2")
    plt.plot(t_plot_grau3, p3_plot, "--", label="Polinômio interpolador de grau 3")

    plt.plot(
        valor_alvo,
        valor_estimado,
        "s",
        markersize=8,
        label=f"Estimativa grau 2: {valor_estimado:.5f} m"
    )

    plt.plot(
        valor_alvo,
        valor_estimado_grau3,
        "^",
        markersize=8,
        label=f"Estimativa grau 3: {valor_estimado_grau3:.5f} m"
    )

    plt.axvline(valor_alvo, linestyle=":", linewidth=1)

    plt.title("Comparação entre interpolações locais de grau 2 e grau 3")
    plt.xlabel("Tempo t (s)")
    plt.ylabel("Altura y(t) (m)")
    plt.grid(True)
    plt.legend(framealpha=1)
    plt.tight_layout()

    nome_arquivo = "projetil_interpolacao_newton_grau2_grau3.eps"
    plt.savefig(nome_arquivo, format="eps")
    plt.show()


# ============================================================
# EXECUÇÃO DO EXERCÍCIO 4
# ============================================================

imprimir_titulo("EXERCÍCIO 4 - PROJÉTIL E DIFERENÇAS DIVIDIDAS")

# ------------------------------------------------------------
# 1) Dados do problema
# ------------------------------------------------------------

imprimir_titulo("1) DADOS DO PROBLEMA")
imprimir_vetor_dados(t, y)

print(f"\nValor que se deseja estimar:")
print(f"t = {valor_alvo:.1f} s")


# ------------------------------------------------------------
# 2) Tabela completa de diferenças divididas
# ------------------------------------------------------------

imprimir_titulo("2) TABELA COMPLETA DE DIFERENÇAS DIVIDIDAS")

tabela_completa = diferencas_divididas(t, y)

print("""
A tabela completa foi calculada para analisar o comportamento geral das
diferenças divididas. Entretanto, como o objetivo é estimar a altura apenas
em t = 5.5 s, a construção do polinômio será feita com pontos locais.
""")

imprimir_tabela_diferencas(t, tabela_completa)


# ------------------------------------------------------------
# 3) Escolha dos pontos locais
# ------------------------------------------------------------

imprimir_titulo("3) ESCOLHA DOS PONTOS LOCAIS")

print("""
O valor desejado é t = 5.5 s, que está entre t = 5 e t = 6.

Para o polinômio principal de grau 2, foram escolhidos os pontos:
t = 5, 6 e 7.

Essa escolha mantém a interpolação local e permite capturar a redução dos
acréscimos da altura após t = 5.
""")

x_local_grau2 = np.array([5, 6, 7], dtype=float)
y_local_grau2 = np.array([4.93, 5.23, 5.38], dtype=float)

x_local_grau3 = np.array([5, 6, 7, 8], dtype=float)
y_local_grau3 = np.array([4.93, 5.23, 5.38, 5.43], dtype=float)


# ------------------------------------------------------------
# 4) Polinômio local de grau 2
# ------------------------------------------------------------

imprimir_titulo("4) POLINÔMIO LOCAL DE GRAU 2")

print("Pontos utilizados no polinômio de grau 2:")
for xi, yi in zip(x_local_grau2, y_local_grau2):
    print(f"({xi:.1f}, {yi:.5f})")

tabela_grau2 = diferencas_divididas(x_local_grau2, y_local_grau2)

imprimir_titulo("4.1) TABELA LOCAL DE DIFERENÇAS DIVIDIDAS - GRAU 2")
imprimir_tabela_diferencas(x_local_grau2, tabela_grau2)

a0_g2 = tabela_grau2[0, 0]
a1_g2 = tabela_grau2[0, 1]
a2_g2 = tabela_grau2[0, 2]

valor_estimado_grau2 = avaliar_newton(
    x_pontos=x_local_grau2,
    tabela=tabela_grau2,
    valor=valor_alvo,
    grau=2
)

imprimir_titulo("4.2) COEFICIENTES E AVALIAÇÃO DO POLINÔMIO DE GRAU 2")

print("Coeficientes de Newton:")
print(f"f[5]     = {a0_g2:.10f}")
print(f"f[5,6]   = {a1_g2:.10f}")
print(f"f[5,6,7] = {a2_g2:.10f}")

print("\nForma de Newton:")
print(
    f"p2(t) = {a0_g2:.10f} "
    f"+ ({a1_g2:.10f})(t - 5) "
    f"+ ({a2_g2:.10f})(t - 5)(t - 6)"
)

print(f"\nAvaliação em t = {valor_alvo:.1f}:")
print(f"p2({valor_alvo:.1f}) = {valor_estimado_grau2:.10f} m")


# ------------------------------------------------------------
# 5) Polinômio local de grau 3
# ------------------------------------------------------------

imprimir_titulo("5) POLINÔMIO LOCAL DE GRAU 3 PARA COMPARAÇÃO")

print("Pontos utilizados no polinômio de grau 3:")
for xi, yi in zip(x_local_grau3, y_local_grau3):
    print(f"({xi:.1f}, {yi:.5f})")

tabela_grau3 = diferencas_divididas(x_local_grau3, y_local_grau3)

imprimir_titulo("5.1) TABELA LOCAL DE DIFERENÇAS DIVIDIDAS - GRAU 3")
imprimir_tabela_diferencas(x_local_grau3, tabela_grau3)

a0_g3 = tabela_grau3[0, 0]
a1_g3 = tabela_grau3[0, 1]
a2_g3 = tabela_grau3[0, 2]
a3_g3 = tabela_grau3[0, 3]

valor_estimado_grau3 = avaliar_newton(
    x_pontos=x_local_grau3,
    tabela=tabela_grau3,
    valor=valor_alvo,
    grau=3
)

imprimir_titulo("5.2) COEFICIENTES E AVALIAÇÃO DO POLINÔMIO DE GRAU 3")

print("Coeficientes de Newton:")
print(f"f[5]       = {a0_g3:.10f}")
print(f"f[5,6]     = {a1_g3:.10f}")
print(f"f[5,6,7]   = {a2_g3:.10f}")
print(f"f[5,6,7,8] = {a3_g3:.10f}")

print("\nForma de Newton:")
print(
    f"p3(t) = {a0_g3:.10f} "
    f"+ ({a1_g3:.10f})(t - 5) "
    f"+ ({a2_g3:.10f})(t - 5)(t - 6) "
    f"+ ({a3_g3:.10f})(t - 5)(t - 6)(t - 7)"
)

print(f"\nAvaliação em t = {valor_alvo:.1f}:")
print(f"p3({valor_alvo:.1f}) = {valor_estimado_grau3:.10f} m")


# ------------------------------------------------------------
# 6) Comparação final
# ------------------------------------------------------------

imprimir_titulo("6) COMPARAÇÃO FINAL ENTRE GRAU 2 E GRAU 3")

diferenca_grau2_grau3 = abs(valor_estimado_grau3 - valor_estimado_grau2)

print(f"p2({valor_alvo:.1f}) = {valor_estimado_grau2:.10f} m")
print(f"p3({valor_alvo:.1f}) = {valor_estimado_grau3:.10f} m")
print(f"Diferença absoluta = {diferenca_grau2_grau3:.10f} m")

# ------------------------------------------------------------
# 7) Geração do gráfico
# ------------------------------------------------------------

imprimir_titulo("7) GERAÇÃO DO GRÁFICO")

gerar_grafico(
    t=t,
    y=y,
    x_local=x_local_grau2,
    coeficientes=(a0_g2, a1_g2, a2_g2),
    valor_alvo=valor_alvo,
    valor_estimado=valor_estimado_grau2
)
