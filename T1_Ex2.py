# Dicente Sophia Lafis Formiga, GRR 20246031
import numpy as np

# ============================================================
# EXERCÍCIO 2 - CÁLCULO NUMÉRICO
# Sistema não linear
# ============================================================

np.set_printoptions(precision=10, suppress=True)

# ============================================================
# FUNÇÃO F E JACOBIANA EXATA
# ============================================================

def F(x):
    """
    Define o sistema não linear F(x) = 0.
    """
    x1 = x[0]
    x2 = x[1]

    return np.array([
        10 * (x2 - x1**2),
        1 - x1
    ], dtype=float)


def J(x):
    """
    Jacobiana exata do sistema.
    """
    x1 = x[0]

    return np.array([
        [-20 * x1, 10],
        [-1,       0]
    ], dtype=float)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def norma_F(x):
    """
    Calcula ||F(x)||_2.
    """
    return np.linalg.norm(F(x), 2)


def erro_relativo(x, x_exata):
    """
    Calcula o erro relativo em relação à solução exata.
    """
    return np.linalg.norm(x - x_exata, 2) / np.linalg.norm(x_exata, 2)


def imprimir_titulo(titulo):
    print("\n" + "=" * 90)
    print(titulo)
    print("=" * 90)


def imprimir_historico(nome_metodo, historico):
    """
    Imprime a evolução das iterações de um método.
    """
    print(f"\n{nome_metodo}")
    print("-" * 90)
    print(f"{'Iteração':>10} | {'x1':>18} | {'x2':>18} | {'||F(x)||_2':>18}")
    print("-" * 90)

    for item in historico:
        k = item["iteracao"]
        x = item["x"]
        norma = item["norma_F"]

        print(f"{k:>10} | {x[0]:>18.10f} | {x[1]:>18.10f} | {norma:>18.10e}")

    print("-" * 90)


def imprimir_resumo(resultados, x_exata):
    """
    Imprime uma tabela final comparando todos os métodos.
    """
    print("\nRESUMO FINAL")
    print("-" * 115)
    print(
        f"{'Método':<40} | "
        f"{'Iterações':>10} | "
        f"{'x1 final':>15} | "
        f"{'x2 final':>15} | "
        f"{'||F(x)||_2':>15} | "
        f"{'Erro relativo':>15}"
    )
    print("-" * 115)

    for nome, dados in resultados.items():
        x_final = dados["x"]
        iteracoes = dados["iteracoes"]
        norma_final = dados["norma_final"]
        erro = erro_relativo(x_final, x_exata)

        print(
            f"{nome:<40} | "
            f"{iteracoes:>10} | "
            f"{x_final[0]:>15.10f} | "
            f"{x_final[1]:>15.10f} | "
            f"{norma_final:>15.6e} | "
            f"{erro:>15.6e}"
        )

    print("-" * 115)

def imprimir_matrizes_B(historico_B):
    """
    Imprime as matrizes B_k usadas no método de Broyden.
    """
    print("\nMATRIZES B_k DO MÉTODO DE BROYDEN")
    print("-" * 90)

    for item in historico_B:
        k = item["iteracao"]
        B = item["B"]

        print(f"\nB_{k}:")
        print(f"[[{B[0,0]:>14.10f}  {B[0,1]:>14.10f}]")
        print(f" [{B[1,0]:>14.10f}  {B[1,1]:>14.10f}]]")

    print("-" * 90)


# ============================================================
# MÉTODO DE NEWTON CLÁSSICO
# ============================================================

def newton_classico(x0, epsilon=1e-4, max_iter=100):
    """
    Método de Newton clássico.
    """
    x = x0.copy().astype(float)

    historico = [{
        "iteracao": 0,
        "x": x.copy(),
        "norma_F": norma_F(x)
    }]

    for k in range(1, max_iter + 1):
        # Resolve J(x) s = -F(x)
        s = np.linalg.solve(J(x), -F(x))

        # Atualiza a solução
        x = x + s

        norma = norma_F(x)

        historico.append({
            "iteracao": k,
            "x": x.copy(),
            "norma_F": norma
        })

        if norma < epsilon:
            return x, k, norma, historico

    return x, max_iter, norma_F(x), historico


# ============================================================
# MÉTODO DE NEWTON MODIFICADO
# ============================================================

def newton_modificado(x0, epsilon=1e-4, max_iter=100):
    """
    Método de Newton modificado.
    """
    x = x0.copy().astype(float)

    J0 = J(x0)

    historico = [{
        "iteracao": 0,
        "x": x.copy(),
        "norma_F": norma_F(x)
    }]

    for k in range(1, max_iter + 1):
        # Resolve J(x0) s = -F(x)
        s = np.linalg.solve(J0, -F(x))

        # Atualiza a solução
        x = x + s

        norma = norma_F(x)

        historico.append({
            "iteracao": k,
            "x": x.copy(),
            "norma_F": norma
        })

        if norma < epsilon:
            return x, k, norma, historico

    return x, max_iter, norma_F(x), historico


# ============================================================
# QUASE-NEWTON POR DIFERENÇAS FINITAS
# ============================================================

def jacobiana_diferencas_finitas(x):
    """
    Aproxima a Jacobiana por diferenças finitas progressivas.

    Para cada variável x_j, usa-se:

        h_j = sqrt(eps_maquina) * max(1, |x_j|)
    """
    x = np.array(x, dtype=float)

    n = len(x)
    B = np.zeros((n, n), dtype=float)

    F_x = F(x)
    eps_maquina = np.finfo(float).eps

    for j in range(n):
        h_j = np.sqrt(eps_maquina) * max(1.0, abs(x[j]))

        x_perturbado = x.copy()
        x_perturbado[j] = x_perturbado[j] + h_j

        B[:, j] = (F(x_perturbado) - F_x) / h_j

    return B


def quase_newton_diferencas_finitas(x0, epsilon=1e-4, max_iter=100):
    """
    Método quase-Newton usando aproximação da Jacobiana por diferenças finitas.
    """
    x = np.array(x0, dtype=float)

    historico = [{
        "iteracao": 0,
        "x": x.copy(),
        "norma_F": norma_F(x)
    }]

    for k in range(1, max_iter + 1):
        B = jacobiana_diferencas_finitas(x)

        s = np.linalg.solve(B, -F(x))

        x = x + s

        norma = norma_F(x)

        historico.append({
            "iteracao": k,
            "x": x.copy(),
            "norma_F": norma
        })

        if norma < epsilon:
            return x, k, norma, historico

    return x, max_iter, norma_F(x), historico


# ============================================================
# QUASE-NEWTON COM BROYDEN
# ============================================================

def quase_newton_broyden(x0, epsilon=1e-4, max_iter=100):
    """
    Método quase-Newton com atualização de Broyden bom.
    """
    x = x0.copy().astype(float)

    B = J(x)

    historico = [{
        "iteracao": 0,
        "x": x.copy(),
        "norma_F": norma_F(x)
    }]

    historico_B = [{
        "iteracao": 0,
        "B": B.copy()
    }]

    for k in range(1, max_iter + 1):
        F_antigo = F(x)

        # Resolve Bk s = -F(xk)
        s = np.linalg.solve(B, -F_antigo)

        # Atualiza x
        x_novo = x + s

        F_novo = F(x_novo)
        norma = np.linalg.norm(F_novo, 2)

        historico.append({
            "iteracao": k,
            "x": x_novo.copy(),
            "norma_F": norma
        })

        if norma < epsilon:
            return x_novo, k, norma, historico, historico_B

        # Atualização de Broyden
        y = F_novo - F_antigo

        denominador = np.dot(s, s)

        if abs(denominador) < 1e-15:
            raise ValueError("Denominador muito pequeno na atualização de Broyden.")

        B = B + np.outer((y - B @ s), s) / denominador

        historico_B.append({
            "iteracao": k,
            "B": B.copy()
        })

        x = x_novo

    return x, max_iter, norma_F(x), historico, historico_B


# ============================================================
# EXECUÇÃO DO EXERCÍCIO 2
# ============================================================

imprimir_titulo("EXERCÍCIO 2 - SISTEMA NÃO LINEAR")

x0 = np.array([-1.2, 1.0], dtype=float)
epsilon = 1e-4
x_exata = np.array([1.0, 1.0], dtype=float)

print(f"Chute inicial: x0 = ({x0[0]:.4f}, {x0[1]:.4f})")
print(f"Tolerância: epsilon = {epsilon:.1e}")
print(f"Solução exata esperada: x* = ({x_exata[0]:.4f}, {x_exata[1]:.4f})")
print(f"Norma inicial ||F(x0)||_2 = {norma_F(x0):.10e}")


# Newton clássico
x_newton, it_newton, norma_newton, hist_newton = newton_classico(
    x0,
    epsilon=epsilon
)

# Newton modificado
x_modificado, it_modificado, norma_modificado, hist_modificado = newton_modificado(
    x0,
    epsilon=epsilon
)

# Quase-Newton por diferenças finitas
x_fd, it_fd, norma_fd, hist_fd = quase_newton_diferencas_finitas(
    x0,
    epsilon=epsilon
)

# Quase-Newton com Broyden
x_broyden, it_broyden, norma_broyden, hist_broyden, hist_B_broyden = quase_newton_broyden(
    x0,
    epsilon=epsilon
)


# ============================================================
# IMPRESSÃO DOS HISTÓRICOS
# ============================================================

imprimir_titulo("HISTÓRICO DAS ITERAÇÕES")

imprimir_historico("Método de Newton clássico", hist_newton)
imprimir_historico("Método de Newton modificado", hist_modificado)
imprimir_historico("Quase-Newton por diferenças finitas", hist_fd)
imprimir_historico("Quase-Newton com Broyden", hist_broyden)
imprimir_matrizes_B(hist_B_broyden)

# ============================================================
# RESUMO FINAL
# ============================================================

resultados = {
    "Newton clássico": {
        "x": x_newton,
        "iteracoes": it_newton,
        "norma_final": norma_newton
    },
    "Newton modificado": {
        "x": x_modificado,
        "iteracoes": it_modificado,
        "norma_final": norma_modificado
    },
    "Quase-Newton diferenças finitas": {
        "x": x_fd,
        "iteracoes": it_fd,
        "norma_final": norma_fd
    },
    "Quase-Newton Broyden": {
        "x": x_broyden,
        "iteracoes": it_broyden,
        "norma_final": norma_broyden
    }
}

imprimir_titulo("COMPARAÇÃO FINAL DOS MÉTODOS")
imprimir_resumo(resultados, x_exata)
