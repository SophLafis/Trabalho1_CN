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


def coeficientes_newton_grau2(x, y):
    """
    Calcula manualmente os coeficientes do polinômio de Newton de grau 2.
    """
    x0, x1, x2 = x
    y0, y1, y2 = y

    f_x0 = y0
    f_x0_x1 = (y1 - y0) / (x1 - x0)
    f_x1_x2 = (y2 - y1) / (x2 - x1)
    f_x0_x1_x2 = (f_x1_x2 - f_x0_x1) / (x2 - x0)

    return f_x0, f_x0_x1, f_x0_x1_x2


def imprimir_polinomio_grau2(x_local, coeficientes, valor_alvo):
    """
    Imprime o polinômio de grau 2 em forma de Newton e calcula o valor no ponto desejado.
    """
    x0, x1, x2 = x_local
    a0, a1, a2 = coeficientes

    valor = a0 + a1 * (valor_alvo - x0) + a2 * (valor_alvo - x0) * (valor_alvo - x1)

    print("\nPolinômio interpolador escolhido")
    print("-" * 90)

    print("Foi escolhido um polinômio de grau 2 usando os pontos:")
    print(f"({x0:.1f}, {a0:.5f}), ({x1:.1f}, {y_local[1]:.5f}), ({x2:.1f}, {y_local[2]:.5f})")

    print("\nCoeficientes de Newton:")
    print(f"f[{x0:.0f}] = {a0:.10f}")
    print(f"f[{x0:.0f},{x1:.0f}] = {a1:.10f}")
    print(f"f[{x0:.0f},{x1:.0f},{x2:.0f}] = {a2:.10f}")

    print("\nForma de Newton:")
    print(
        f"p2(t) = {a0:.10f} "
        f"+ ({a1:.10f})(t - {x0:.1f}) "
        f"+ ({a2:.10f})(t - {x0:.1f})(t - {x1:.1f})"
    )

    print(f"\nAvaliação em t = {valor_alvo:.1f}:")
    print(f"p2({valor_alvo:.1f}) = {valor:.10f}")

    print(f"\nAltura estimada:")
    print(f"y({valor_alvo:.1f}) ≈ {valor:.10f} m")

    print("-" * 90)

    return valor


# ============================================================
# GRÁFICO
# ============================================================

def gerar_grafico(t, y, x_local, coeficientes, valor_alvo, valor_estimado):
    """
    Gera um gráfico dos dados do projétil e do polinômio local escolhido.
    """
    a0, a1, a2 = coeficientes
    x0, x1, x2 = x_local

    t_plot = np.linspace(3.5, 6.5, 300)

    p2_plot = (
        a0
        + a1 * (t_plot - x0)
        + a2 * (t_plot - x0) * (t_plot - x1)
    )

    plt.figure(figsize=(9, 5.5))

    plt.plot(t, y, "o", label="Dados observados")
    plt.plot(t_plot, p2_plot, "-", label="Polinômio interpolador de grau 2")
    plt.plot(valor_alvo, valor_estimado, "s", markersize=8, label=f"Estimativa em t = {valor_alvo}")

    plt.title("Estimativa da altura do projétil por interpolação de Newton")
    plt.xlabel("Tempo t (s)")
    plt.ylabel("Altura y(t) (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    nome_arquivo = "projetil_interpolacao_newton.eps"
    plt.savefig(nome_arquivo, format="eps")
    plt.show()

    print(f"\nGráfico salvo como: {nome_arquivo}")


# ============================================================
# EXECUÇÃO DO EXERCÍCIO 4
# ============================================================

imprimir_titulo("EXERCÍCIO 4 - PROJÉTIL E DIFERENÇAS DIVIDIDAS")

imprimir_vetor_dados(t, y)

# Tabela completa de diferenças divididas
tabela_completa = diferencas_divididas(t, y)

imprimir_titulo("TABELA COMPLETA DE DIFERENÇAS DIVIDIDAS")
imprimir_tabela_diferencas(t, tabela_completa)


# ============================================================
# ESCOLHA DO POLINÔMIO LOCAL
# ============================================================

imprimir_titulo("ESCOLHA DO POLINÔMIO PARA ESTIMAR y(5.5)")

print("""
O valor desejado é t = 5.5 s.

Os pontos escolhidos foram:
t = 5, 6 e 7.
""")

# Pontos locais escolhidos para o polinômio de grau 2
x_local = np.array([5, 6, 7], dtype=float)
y_local = np.array([4.93, 5.23, 5.38], dtype=float)

# Tabela local
tabela_local = diferencas_divididas(x_local, y_local)

imprimir_titulo("TABELA LOCAL DE DIFERENÇAS DIVIDIDAS")
imprimir_tabela_diferencas(x_local, tabela_local)

# Coeficientes e avaliação
coeficientes = coeficientes_newton_grau2(x_local, y_local)

valor_estimado = imprimir_polinomio_grau2(
    x_local=x_local,
    coeficientes=coeficientes,
    valor_alvo=valor_alvo
)

# Conferência usando a função geral de Newton
valor_conferencia = avaliar_newton(
    x_pontos=x_local,
    tabela=tabela_local,
    valor=valor_alvo,
    grau=2
)

print("\nConferência pelo avaliador geral de Newton:")
print(f"p2({valor_alvo:.1f}) = {valor_conferencia:.10f}")

# Gráfico
gerar_grafico(
    t=t,
    y=y,
    x_local=x_local,
    coeficientes=coeficientes,
    valor_alvo=valor_alvo,
    valor_estimado=valor_estimado
)