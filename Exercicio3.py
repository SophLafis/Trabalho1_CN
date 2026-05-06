# Dicente Sophia Lafis Formiga, GRR 20246031
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# EXERCÍCIO 3 - FENÔMENO DE RUNGE
# ============================================================

# ============================================================
# FUNÇÃO DO PROBLEMA
# ============================================================

def f(x):
    """
    Função de Runge definida no enunciado.
    """
    return 1 / (1 + 25 * x**2)


# ============================================================
# DIFERENÇAS DIVIDIDAS DE NEWTON
# ============================================================

def diferencas_divididas(x, y):
    """
    Constrói a tabela de diferenças divididas.
    """
    n = len(x)
    tabela = np.zeros((n, n), dtype=float)

    tabela[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            tabela[i, j] = (tabela[i + 1, j - 1] - tabela[i, j - 1]) / (x[i + j] - x[i])

    return tabela


def avaliar_newton(x_pontos, tabela, valor):
    """
    Avalia o polinômio interpolador de Newton em um ponto..
    """
    n = len(x_pontos)

    resultado = tabela[0, 0]
    produto = 1.0

    for j in range(1, n):
        produto *= (valor - x_pontos[j - 1])
        resultado += tabela[0, j] * produto

    return resultado


def interpolacao_newton(x_pontos, y_pontos, x_avaliar):
    """
    Calcula os valores do polinômio interpolador de Newton nos pontos x_avaliar.
    """
    tabela = diferencas_divididas(x_pontos, y_pontos)

    y_newton = []

    for valor in x_avaliar:
        y_newton.append(avaliar_newton(x_pontos, tabela, valor))

    return np.array(y_newton), tabela


# ============================================================
# SPLINE LINEAR
# ============================================================

def spline_linear(x_pontos, y_pontos, x_avaliar):
    """
    Calcula a spline linear interpolante.
    """
    y_avaliado = []

    for x in x_avaliar:
        # Caso o ponto seja exatamente o último ponto da malha
        if np.isclose(x, x_pontos[-1]):
            y_avaliado.append(y_pontos[-1])
            continue

        # Encontra o intervalo [x_k, x_{k+1}] que contém x
        k = np.searchsorted(x_pontos, x) - 1
        k = max(0, min(k, len(x_pontos) - 2))

        h = x_pontos[k + 1] - x_pontos[k]
        t = (x - x_pontos[k]) / h

        y = y_pontos[k] + t * (y_pontos[k + 1] - y_pontos[k])
        y_avaliado.append(y)

    return np.array(y_avaliado)


# ============================================================
# SPLINE CÚBICA NATURAL
# ============================================================

def spline_cubica_natural(x_pontos, y_pontos, x_avaliar):
    """
    Calcula a spline cúbica natural.
    """
    n = len(x_pontos)
    h = np.diff(x_pontos)

    # Sistema linear para encontrar as segundas derivadas g_i
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    # Condições naturais nas extremidades
    A[0, 0] = 1
    A[-1, -1] = 1

    # Equações internas
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

        b[i] = 6 * (
            (y_pontos[i + 1] - y_pontos[i]) / h[i]
            - (y_pontos[i] - y_pontos[i - 1]) / h[i - 1]
        )

    g = np.linalg.solve(A, b)

    # Avaliação da spline
    y_avaliado = []

    for x in x_avaliar:
        # Caso o ponto seja exatamente o último ponto da malha
        if np.isclose(x, x_pontos[-1]):
            k = n - 2
        else:
            k = np.searchsorted(x_pontos, x) - 1
            k = max(0, min(k, n - 2))

        dx = x - x_pontos[k]

        a = y_pontos[k]
        b_coef = (
            (y_pontos[k + 1] - y_pontos[k]) / h[k]
            - h[k] * (2 * g[k] + g[k + 1]) / 6
        )
        c = g[k] / 2
        d = (g[k + 1] - g[k]) / (6 * h[k])

        y = a + b_coef * dx + c * dx**2 + d * dx**3
        y_avaliado.append(y)

    return np.array(y_avaliado), g

# ============================================================
# IMPRESSÕES ORGANIZADAS
# ============================================================

def imprimir_titulo(titulo):
    print("\n" + "=" * 100)
    print(titulo)
    print("=" * 100)


def imprimir_tabela_erros(resultados):
    print("\nTabela de erros máximos")
    print("-" * 100)
    print(
        f"{'n':>5} | "
        f"{'max |f(zi)-pn(zi)|':>25} | "
        f"{'max |f(zi)-S1(zi)|':>25} | "
        f"{'max |f(zi)-S3(zi)|':>25}"
    )
    print("-" * 100)

    for linha in resultados:
        print(
            f"{linha['n']:>5} | "
            f"{linha['erro_newton']:>25.10f} | "
            f"{linha['erro_linear']:>25.10f} | "
            f"{linha['erro_cubica']:>25.10f}"
        )

    print("-" * 100)

# ============================================================
# GRÁFICOS
# ============================================================

def gerar_grafico(n, x_pontos, y_pontos, x_plot, y_real, y_newton, y_linear, y_cubica):
    """
    Gera e salva o gráfico comparando a função original com os três métodos de interpolação.
    """
    plt.figure(figsize=(9, 5.5))

    plt.plot(x_plot, y_real, label="Função original f(x)", linewidth=2)
    plt.plot(x_plot, y_newton, "--", label=f"Polinômio de Newton p_{n}(x)")
    plt.plot(x_plot, y_linear, "-.", label="Spline linear S1(x)")
    plt.plot(x_plot, y_cubica, ":", label="Spline cúbica natural S3(x)")
    plt.plot(x_pontos, y_pontos, "o", label="Pontos de interpolação")

    plt.title(f"Fenômeno de Runge para n = {n}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    nome_arquivo = f"runge_n_{n}.eps"
    plt.savefig(nome_arquivo, format="eps")
    plt.show()

    print(f"Gráfico salvo como: {nome_arquivo}")


# ============================================================
# EXECUÇÃO DO EXERCÍCIO 3
# ============================================================

imprimir_titulo("EXERCÍCIO 3 - FENÔMENO DE RUNGE")

# Intervalo do problema
a = -5
b = 5

# Valores de n pedidos no enunciado
valores_n = [5, 10, 20]

# Pontos z_i pedidos no enunciado:
# z_i = -5 + 0.2 i, i = 0, 1, ..., 50
z = np.array([a + 0.2 * i for i in range(51)], dtype=float)

# Pontos mais densos apenas para desenhar gráficos suaves
x_plot = np.linspace(a, b, 1000)
y_plot_real = f(x_plot)

resultados = []

for n in valores_n:
    imprimir_titulo(f"RESULTADOS PARA n = {n}")

    # n + 1 pontos igualmente espaçados em [-5, 5]
    x_pontos = np.linspace(a, b, n + 1)
    y_pontos = f(x_pontos)

    # Avaliação nos pontos z_i para cálculo dos erros
    y_real_z = f(z)

    y_newton_z, tabela_newton = interpolacao_newton(x_pontos, y_pontos, z)
    y_linear_z = spline_linear(x_pontos, y_pontos, z)
    y_cubica_z, g = spline_cubica_natural(x_pontos, y_pontos, z)

    # Erros máximos
    erro_newton = np.max(np.abs(y_real_z - y_newton_z))
    erro_linear = np.max(np.abs(y_real_z - y_linear_z))
    erro_cubica = np.max(np.abs(y_real_z - y_cubica_z))

    resultados.append({
        "n": n,
        "erro_newton": erro_newton,
        "erro_linear": erro_linear,
        "erro_cubica": erro_cubica
    })

    print(f"Erro máximo Newton:        {erro_newton:.10f}")
    print(f"Erro máximo spline linear: {erro_linear:.10f}")
    print(f"Erro máximo spline cúbica: {erro_cubica:.10f}")

    # Avaliação em uma malha densa apenas para gráfico
    y_newton_plot, _ = interpolacao_newton(x_pontos, y_pontos, x_plot)
    y_linear_plot = spline_linear(x_pontos, y_pontos, x_plot)
    y_cubica_plot, _ = spline_cubica_natural(x_pontos, y_pontos, x_plot)

    gerar_grafico(
        n=n,
        x_pontos=x_pontos,
        y_pontos=y_pontos,
        x_plot=x_plot,
        y_real=y_plot_real,
        y_newton=y_newton_plot,
        y_linear=y_linear_plot,
        y_cubica=y_cubica_plot
    )


# ============================================================
# RESUMO FINAL
# ============================================================

imprimir_titulo("RESUMO FINAL DOS ERROS")

imprimir_tabela_erros(resultados)