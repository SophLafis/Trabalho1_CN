# Dicente Sophia Lafis Formiga, GRR 20246031
import numpy as np

# ============================================================
# EXERCÍCIO 1 - CÁLCULO NUMÉRICO
# Sistema linear Ax = b
# ============================================================

A = np.array([
    [1000, -1,    0,    0,   0,   0,   0,  0,  0,  0],
    [-1,  500,   -1,    0,   0,   0,   0,  0,  0,  0],
    [0,   -1,   200,   -1,   0,   0,   0,  0,  0,  0],
    [0,    0,    -1,  100,  -1,   0,   0,  0,  0,  0],
    [0,    0,     0,   -1,  50,  -1,   0,  0,  0,  0],
    [0,    0,     0,    0,  -1,  20,  -1,  0,  0,  0],
    [0,    0,     0,    0,   0,  -1,  10, -1,  0,  0],
    [0,    0,     0,    0,   0,   0,  -1,  5, -1,  0],
    [0,    0,     0,    0,   0,   0,   0, -1,  2, -1],
    [0,    0,     0,    0,   0,   0,   0,  0, -1,  1]
], dtype=float)

b = np.ones(10, dtype=float)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def residuo_relativo(A, b, x):
    """
    Calcula o resíduo relativo
    Esse valor mede o quanto a solução aproximada x satisfaz o sistema Ax = b.
    """
    r = b - A @ x
    return np.linalg.norm(r, 2) / np.linalg.norm(b, 2)


def verificar_simetria(A):
    """
    Verifica se A é simétrica, isto é, se A = A^T.
    """
    return np.allclose(A, A.T)


def verificar_definida_positiva(A):
    """
    Uma matriz simétrica é definida positiva se todos os seus autovalores são positivos.
    """
    autovalores = np.linalg.eigvalsh(A)
    return np.all(autovalores > 0), autovalores

# ============================================================
# LETRA (a): NÚMERO DE CONDICIONAMENTO
# ============================================================

def calcular_condicionamento(A):
    """
    Calcula o número de condicionamento na norma 2.
    """
    cond_2 = np.linalg.cond(A, 2)

    autovalores = np.linalg.eigvalsh(A)
    lambda_min = np.min(autovalores)
    lambda_max = np.max(autovalores)

    return cond_2, lambda_min, lambda_max

# ============================================================
# LETRA (b): FATORAÇÃO LU
# Implementação simples com pivoteamento parcial
# ============================================================

def lu_decomposition(A):
    """
    Fatoração LU com pivoteamento parcial.
    """
    A = A.copy().astype(float)
    n = A.shape[0]

    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    for k in range(n):
        # Escolhe o maior pivô em módulo na coluna k
        pivot = np.argmax(np.abs(U[k:, k])) + k

        if np.isclose(U[pivot, k], 0):
            raise ValueError("Matriz singular ou quase singular.")

        # Troca linhas em U e P
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]

            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        # Eliminação de Gauss
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, :] = U[i, :] - L[i, k] * U[k, :]

    np.fill_diagonal(L, 1.0)

    return P, L, U


def substituicao_progressiva(L, b):
    """
    Resolve Ly = b, onde L é triangular inferior.
    """
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        soma = np.dot(L[i, :i], y[:i])
        y[i] = (b[i] - soma) / L[i, i]

    return y


def substituicao_regressiva(U, y):
    """
    Resolve Ux = y, onde U é triangular superior.
    """
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        soma = np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = (y[i] - soma) / U[i, i]

    return x


def resolver_lu(A, b):
    """
    Resolve Ax = b usando fatoração LU com pivoteamento parcial.
    """
    P, L, U = lu_decomposition(A)
    Pb = P @ b

    y = substituicao_progressiva(L, Pb)
    x = substituicao_regressiva(U, y)

    return x


# ============================================================
# LETRA (b): FATORAÇÃO DE CHOLESKY
# ============================================================

def resolver_cholesky(A, b):
    """
    Resolve Ax = b por Cholesky.
    """
    L = np.linalg.cholesky(A)

    y = substituicao_progressiva(L, b)
    x = substituicao_regressiva(L.T, y)

    return x


# ============================================================
# LETRA (c): MÉTODO DE JACOBI
# ============================================================

def jacobi(A, b, x0, tol=1e-8, max_iter=10000):
    """
    Método de Jacobi.
    """
    D = np.diag(A)
    R = A - np.diagflat(D)

    x = x0.copy()

    historico_residuos = []

    for k in range(1, max_iter + 1):
        x = (b - R @ x) / D

        res_rel = residuo_relativo(A, b, x)
        historico_residuos.append(res_rel)

        if res_rel < tol:
            return x, k, res_rel, historico_residuos

    return x, max_iter, residuo_relativo(A, b, x), historico_residuos


# ============================================================
# LETRA (c): MÉTODO DE GAUSS-SEIDEL
# ============================================================

def gauss_seidel(A, b, x0, tol=1e-8, max_iter=10000):
    """
    Método de Gauss-Seidel.
    """
    n = len(b)
    x = x0.copy()

    historico_residuos = []

    for k in range(1, max_iter + 1):
        for i in range(n):
            soma_antes = np.dot(A[i, :i], x[:i])
            soma_depois = np.dot(A[i, i + 1:], x[i + 1:])

            x[i] = (b[i] - soma_antes - soma_depois) / A[i, i]

        res_rel = residuo_relativo(A, b, x)
        historico_residuos.append(res_rel)

        if res_rel < tol:
            return x, k, res_rel, historico_residuos

    return x, max_iter, residuo_relativo(A, b, x), historico_residuos


# ============================================================
# LETRA (c): GRADIENTE CONJUGADO
# ============================================================

def gradiente_conjugado(A, b, x0, tol=1e-8, max_iter=10000):
    """
    Método do Gradiente Conjugado.
    """
    x = x0.copy()

    r = b - A @ x
    p = r.copy()

    rs_old = np.dot(r, r)

    historico_residuos = [np.linalg.norm(r, 2) / np.linalg.norm(b, 2)]

    for k in range(1, max_iter + 1):
        Ap = A @ p

        alpha = rs_old / np.dot(p, Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        res_rel = np.linalg.norm(r, 2) / np.linalg.norm(b, 2)
        historico_residuos.append(res_rel)

        if res_rel < tol:
            return x, k, res_rel, historico_residuos

        rs_new = np.dot(r, r)
        beta = rs_new / rs_old

        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, residuo_relativo(A, b, x), historico_residuos


# ============================================================
# LETRA (d): GRADIENTE CONJUGADO PRECONDICIONADO
# Precondicionador diagonal de Jacobi
# ============================================================

def gradiente_conjugado_precondicionado_jacobi(A, b, x0, tol=1e-8, max_iter=10000):
    """
    Gradiente Conjugado Precondicionado usando precondicionador diagonal de Jacobi.
    """
    x = x0.copy()

    r = b - A @ x

    M_inv = np.diag(1.0 / np.diag(A))

    z = M_inv @ r
    p = z.copy()

    rz_old = np.dot(r, z)

    historico_residuos = [np.linalg.norm(r, 2) / np.linalg.norm(b, 2)]

    for k in range(1, max_iter + 1):
        Ap = A @ p

        alpha = rz_old / np.dot(p, Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        res_rel = np.linalg.norm(r, 2) / np.linalg.norm(b, 2)
        historico_residuos.append(res_rel)

        if res_rel < tol:
            return x, k, res_rel, historico_residuos

        z = M_inv @ r

        rz_new = np.dot(r, z)
        beta = rz_new / rz_old

        p = z + beta * p
        rz_old = rz_new

    return x, max_iter, residuo_relativo(A, b, x), historico_residuos

# ============================================================
# EXECUÇÃO DO EXERCÍCIO
# Importante citar que eu pedi ajuda do chat para os resultados ficarem mais bonitos visualmente no terminal :)
# ============================================================

def imprimir_titulo(titulo):
    print("\n" + "=" * 90)
    print(titulo)
    print("=" * 90)


def imprimir_subtitulo(titulo):
    print("\n" + "-" * 90)
    print(titulo)
    print("-" * 90)


def imprimir_vetor(nome, x):
    print(f"\n{nome}")
    print("-" * 42)
    print(f"{'Índice':>8} | {'Valor':>25}")
    print("-" * 42)

    for i, valor in enumerate(x, start=1):
        print(f"{i:>8} | {valor:>25.10f}")

    print("-" * 42)


def imprimir_comparacao_solucoes(titulo, solucoes):
    nomes = list(solucoes.keys())
    largura = 12 + 23 * len(nomes)

    print(f"\n{titulo}")
    print("-" * largura)

    cabecalho = f"{'Índice':>8} |"
    for nome in nomes:
        cabecalho += f" {nome:>19} |"
    print(cabecalho)

    print("-" * largura)

    n = len(next(iter(solucoes.values())))

    for i in range(n):
        linha = f"{i + 1:>8} |"
        for nome in nomes:
            linha += f" {solucoes[nome][i]:>19.10f} |"
        print(linha)

    print("-" * largura)


def erro_relativo(x, x_referencia):
    return np.linalg.norm(x - x_referencia, 2) / np.linalg.norm(x_referencia, 2)


def imprimir_resumo_metodos(titulo, linhas):
    print(f"\n{titulo}")
    print("-" * 105)
    print(
        f"{'Método':<42} | "
        f"{'Iterações':>10} | "
        f"{'Resíduo relativo':>20} | "
        f"{'Erro vs referência':>20}"
    )
    print("-" * 105)

    for metodo, iteracoes, residuo, erro in linhas:
        iteracoes_txt = "-" if iteracoes is None else str(iteracoes)

        print(
            f"{metodo:<42} | "
            f"{iteracoes_txt:>10} | "
            f"{residuo:>20.10e} | "
            f"{erro:>20.10e}"
        )

    print("-" * 105)


# Verificações iniciais

imprimir_titulo("VERIFICAÇÕES DA MATRIZ")

simetrica = verificar_simetria(A)
def_pos, autovalores = verificar_definida_positiva(A)

print(f"Matriz simétrica:             {simetrica}")
print(f"Matriz definida positiva:     {def_pos}")
print(f"Menor autovalor:              {np.min(autovalores):.10f}")
print(f"Maior autovalor:              {np.max(autovalores):.10f}")


# Letra (a)

imprimir_titulo("LETRA (a) - NÚMERO DE CONDICIONAMENTO")

cond_2, lambda_min, lambda_max = calcular_condicionamento(A)

print(f"Condicionamento na norma 2:   {cond_2:.10f}")
print(f"Menor autovalor:              {lambda_min:.10f}")
print(f"Maior autovalor:              {lambda_max:.10f}")


# Letra (b): métodos diretos

imprimir_titulo("LETRA (b) - MÉTODOS DIRETOS")

x_lu = resolver_lu(A, b)
x_cholesky = resolver_cholesky(A, b)
x_referencia = np.linalg.solve(A, b)

res_lu = residuo_relativo(A, b, x_lu)
res_cholesky = residuo_relativo(A, b, x_cholesky)
res_referencia = residuo_relativo(A, b, x_referencia)

erro_lu = erro_relativo(x_lu, x_referencia)
erro_cholesky = erro_relativo(x_cholesky, x_referencia)
erro_referencia = erro_relativo(x_referencia, x_referencia)

imprimir_resumo_metodos(
    "Resumo dos métodos diretos",
    [
        ("Fatoração LU", None, res_lu, erro_lu),
        ("Fatoração de Cholesky", None, res_cholesky, erro_cholesky),
        ("np.linalg.solve", None, res_referencia, erro_referencia),
    ]
)

imprimir_comparacao_solucoes(
    "Comparação das soluções obtidas pelos métodos diretos",
    {
        "LU": x_lu,
        "Cholesky": x_cholesky,
        "Referência": x_referencia
    }
)

# Letra (c): métodos iterativos

imprimir_titulo("LETRA (c) - MÉTODOS ITERATIVOS")

x0 = np.zeros(len(b))
tol = 1e-8

x_jacobi, it_jacobi, res_jacobi, hist_jacobi = jacobi(A, b, x0, tol)
x_gs, it_gs, res_gs, hist_gs = gauss_seidel(A, b, x0, tol)
x_cg, it_cg, res_cg, hist_cg = gradiente_conjugado(A, b, x0, tol)

erro_jacobi = erro_relativo(x_jacobi, x_referencia)
erro_gs = erro_relativo(x_gs, x_referencia)
erro_cg = erro_relativo(x_cg, x_referencia)

imprimir_resumo_metodos(
    "Resumo dos métodos iterativos",
    [
        ("Jacobi", it_jacobi, res_jacobi, erro_jacobi),
        ("Gauss-Seidel", it_gs, res_gs, erro_gs),
        ("Gradiente Conjugado", it_cg, res_cg, erro_cg),
    ]
)

imprimir_comparacao_solucoes(
    "Comparação das soluções obtidas pelos métodos iterativos",
    {
        "Jacobi": x_jacobi,
        "Gauss-Seidel": x_gs,
        "CG": x_cg,
        "Referência": x_referencia
    }
)

# Letra (d): Gradiente Conjugado Precondicionado

imprimir_titulo("LETRA (d) - GRADIENTE CONJUGADO PRECONDICIONADO")

x_pcg, it_pcg, res_pcg, hist_pcg = gradiente_conjugado_precondicionado_jacobi(A, b, x0, tol)

erro_pcg = erro_relativo(x_pcg, x_referencia)

D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(A)))
A_precond_sim = D_inv_sqrt @ A @ D_inv_sqrt

cond_precond = np.linalg.cond(A_precond_sim, 2)

print(f"Condicionamento original:                         {cond_2:.10f}")
print(f"Condicionamento após precondicionamento diagonal: {cond_precond:.10f}")

imprimir_resumo_metodos(
    "Comparação entre Gradiente Conjugado e PCG",
    [
        ("Gradiente Conjugado", it_cg, res_cg, erro_cg),
        ("Gradiente Conjugado Precondicionado", it_pcg, res_pcg, erro_pcg),
    ]
)

imprimir_comparacao_solucoes(
    "Comparação entre CG, PCG e solução de referência",
    {
        "CG": x_cg,
        "PCG": x_pcg,
        "Referência": x_referencia
    }
)

imprimir_vetor("Solução final de referência obtida pelo código", x_referencia)