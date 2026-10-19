"""
scanline.py

Modulo 4 do TP1: preenchimento de poligonos com o algoritmo Scanline Fill,
implementado pelo grupo. O Matplotlib e usado apenas para mostrar a grade.

Convencao: o centro do pixel (i, j) fica na coordenada inteira (i, j).
Um pixel e pintado se o seu centro esta dentro do poligono.

Uso:
    python scanline.py ../data/exemplo2d.vtk
Gera ../report/figures/scanline_fill.png
"""

import os
import random
import sys
from math import atan2, ceil

import matplotlib.pyplot as plt
import numpy as np

from bresenham import desenhar_grade, rasterizar_arvore
from util import (carregar_modelo_2d, carregar_segmentos_2d,
                  criar_mapeamento, extremos_arvore)


# ---------------------------------------------------------------------------
# Algoritmo
# ---------------------------------------------------------------------------

def scanline_fill(vertices, largura, altura):
    """
    Preenche o poligono dado pela lista de vertices [(x, y), ...].
    Devolve:
      grade        : matriz booleana (altura, largura), True = pixel pintado
      interseccoes : dict {y: [x1, x2, ...]} com os cruzamentos de cada linha
                     (usado para mostrar o funcionamento passo a passo)

    Passos:
      1. Monta a lista de arestas, ignorando as horizontais.
      2. Para cada linha y (scanline), acha as arestas que ela cruza.
      3. Calcula o x de cada cruzamento e ordena.
      4. Pinta entre os pares (1o-2o, 3o-4o, ...): regra par-impar.
    """
    # 1) Arestas guardadas como (y_min, y_max, x em y_min, dx/dy)
    arestas = []
    n = len(vertices)
    for i in range(n):
        (x0, y0), (x1, y1) = vertices[i], vertices[(i + 1) % n]
        if y0 == y1:
            continue                      # horizontal: nao cruza nenhuma scanline
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        arestas.append((y0, y1, x0, (x1 - x0) / (y1 - y0)))

    grade = np.zeros((altura, largura), dtype=bool)
    interseccoes = {}

    ys = [v[1] for v in vertices]
    y_ini = max(0, ceil(min(ys)))
    y_fim = min(altura - 1, ceil(max(ys)) - 1)

    for y in range(y_ini, y_fim + 1):
        # 2) e 3) Cruzamentos da linha y. A aresta conta se y_min <= y < y_max:
        # o intervalo semi-aberto evita contar duas vezes um vertice
        # compartilhado por duas arestas.
        xs = sorted(x0 + (y - ymin) * m
                    for ymin, ymax, x0, m in arestas if ymin <= y < ymax)
        interseccoes[y] = xs

        # 4) Pinta entre os pares. Pixel x e pintado se xa <= x < xb.
        for xa, xb in zip(xs[0::2], xs[1::2]):
            for x in range(max(0, ceil(xa)), min(largura, ceil(xb))):
                grade[y, x] = True

    return grade, interseccoes


# ---------------------------------------------------------------------------
# Verificacao: comparar com um metodo independente (raio horizontal)
# ---------------------------------------------------------------------------

def ponto_dentro(px, py, vertices):
    """Teste par-impar por lancamento de raio para a direita (referencia)."""
    dentro = False
    n = len(vertices)
    for i in range(n):
        (x0, y0), (x1, y1) = vertices[i], vertices[(i + 1) % n]
        if (y0 <= py < y1) or (y1 <= py < y0):
            x_cruz = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            if x_cruz > px:
                dentro = not dentro
    return dentro


def verificar(vertices, largura, altura):
    grade, _ = scanline_fill(vertices, largura, altura)
    for y in range(altura):
        for x in range(largura):
            assert grade[y, x] == ponto_dentro(x, y, vertices), f"pixel ({x},{y})"


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

# Poligono concavo em formato de U (testa a regra par-impar)
POLIGONO_U = [(2.5, 2.5), (21.5, 2.5), (21.5, 17.5), (15.5, 17.5),
              (15.5, 8.5), (8.5, 8.5), (8.5, 17.5), (2.5, 17.5)]

# Regiao de interesse (hexagono) sobre a grade 64x64 da arvore
POLIGONO_ROI = [(18.5, 26.5), (40.5, 22.5), (48.5, 38.5),
                (38.5, 52.5), (20.5, 48.5), (12.5, 36.5)]


def sobrepor_poligono(ax, vertices, grade, cor):
    """Pinta os pixels preenchidos e desenha o contorno real do poligono."""
    for y, x in zip(*np.nonzero(grade)):
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=cor, alpha=0.45, lw=0))
    xs, ys = zip(*(vertices + [vertices[0]]))
    ax.plot(xs, ys, "r-", linewidth=1.3)


def figura(segs, raios, caminho_saida, y_demo=12):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))

    # Painel 1: poligono concavo com uma scanline destacada
    w, h = 24, 20
    grade, inter = scanline_fill(POLIGONO_U, w, h)
    desenhar_grade(a1, w, h)
    sobrepor_poligono(a1, POLIGONO_U, grade, "tab:blue")
    a1.axhline(y_demo, color="green", linewidth=1.5, label=f"scanline y = {y_demo}")
    xs = inter[y_demo]
    a1.plot(xs, [y_demo] * len(xs), "go", markersize=7, markeredgecolor="k")
    a1.set_title(f"Poligono concavo (U): {int(grade.sum())} pixels preenchidos")
    a1.legend(loc="upper right", fontsize=8)

    # Painel 2: regiao de interesse sobre a arvore rasterizada
    n = 64
    arvore, _ = rasterizar_arvore(segs, raios, n)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    a2.imshow(arvore, origin="lower", cmap=cmap, vmin=raios.min(),
              vmax=raios.max(), interpolation="nearest")
    desenhar_grade(a2, n, n)
    roi, _ = scanline_fill(POLIGONO_ROI, n, n)
    sobrepor_poligono(a2, POLIGONO_ROI, roi, "orange")
    a2.set_title(f"Regiao de interesse sobre a arvore: {int(roi.sum())} pixels")

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return inter, y_demo



# ---------------------------------------------------------------------------
# Poligono formado pelos extremos da arvore (raiz + pontas das folhas)
# ---------------------------------------------------------------------------

def ordenar_por_angulo(pontos):
    """
    Ordena os pontos em torno do centroide, pelo angulo.
    Isso liga cada ponto ao seu vizinho angular e gera um poligono simples
    (sem arestas se cruzando), que pode ser concavo.
    Devolve a lista de posicoes na nova ordem.
    """
    cx = sum(p[0] for p in pontos) / len(pontos)
    cy = sum(p[1] for p in pontos) / len(pontos)
    return sorted(range(len(pontos)),
                  key=lambda i: atan2(pontos[i][1] - cy, pontos[i][0] - cx))


def arestas_se_cruzam(a, b, c, d):
    """True se os segmentos ab e cd se cruzam (teste por orientacao)."""
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return int(v > 0) - int(v < 0)
    return orient(a, b, c) != orient(a, b, d) and orient(c, d, a) != orient(c, d, b)


def poligono_simples(vertices):
    """Confere que nenhuma aresta cruza outra aresta nao adjacente."""
    n = len(vertices)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue                  # primeira e ultima sao adjacentes
            if arestas_se_cruzam(vertices[i], vertices[(i + 1) % n],
                                 vertices[j], vertices[(j + 1) % n]):
                return False
    return True


def area_shoelace(v):
    """Area do poligono pela formula do cadarco (para conferir os pixels)."""
    n = len(v)
    return abs(sum(v[i][0] * v[(i + 1) % n][1] - v[(i + 1) % n][0] * v[i][1]
                   for i in range(n))) / 2


def figura_extremos(pontos, indices, raios, caminho_saida, n=64):
    raiz, folhas = extremos_arvore(indices)
    ids = [raiz] + folhas

    # Mesmo mapeamento da arvore rasterizada, mantendo coordenadas fracionarias
    mapear = criar_mapeamento(pontos[indices], n, n)
    pix = [mapear(*pontos[i]) for i in ids]

    ordem = ordenar_por_angulo(pix)
    ids = [ids[k] for k in ordem]
    vertices = [pix[k] for k in ordem]

    assert poligono_simples(vertices), "poligono com arestas se cruzando"
    grade, _ = scanline_fill(vertices, n, n)
    for y in range(n):                    # confere com o metodo de referencia
        for x in range(n):
            assert grade[y, x] == ponto_dentro(x, y, vertices)

    print(f"Raiz: ponto {raiz} | folhas: {folhas}")
    print("Ordem dos vertices (angulo em torno do centroide):")
    for i, (x, y) in zip(ids, vertices):
        print(f"  ponto {i:2d} -> pixel ({x:6.2f}, {y:6.2f})")
    print(f"Pixels preenchidos: {int(grade.sum())} | area exata: "
          f"{area_shoelace(vertices):.1f}")

    # Desenho: arvore + poligono preenchido
    fig, ax = plt.subplots(figsize=(8, 8))
    arvore, _ = rasterizar_arvore(pontos[indices], raios, n)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((1, 1, 1, 0))            # pixels vazios transparentes
    # A arvore fica POR CIMA do preenchimento (zorder), preservando as cores
    img = ax.imshow(arvore, origin="lower", cmap=cmap, vmin=raios.min(),
                    vmax=raios.max(), interpolation="nearest", zorder=1.5)
    desenhar_grade(ax, n, n)
    sobrepor_poligono(ax, vertices, grade, "orange")
    for i, (x, y) in zip(ids, vertices):
        ax.plot(x, y, "ko", markersize=5)
        ax.annotate("raiz" if i == raiz else f"folha {i}", (x, y),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    fig.colorbar(img, ax=ax, shrink=0.7, label="raio")
    ax.set_title("Poligono formado pela raiz e pelas folhas "
                 f"({int(grade.sum())} pixels)")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scanline.py arquivo.vtk")
        sys.exit(1)

    segs, raios = carregar_segmentos_2d(sys.argv[1])
    saida = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "report", "figures")
    os.makedirs(saida, exist_ok=True)

    # 1) Testes contra o metodo de referencia
    verificar(POLIGONO_U, 24, 20)
    verificar(POLIGONO_ROI, 64, 64)
    random.seed(1)
    for _ in range(200):               # poligonos aleatorios (podem ser concavos)
        k = random.randint(3, 8)
        poli = [(random.uniform(0, 30), random.uniform(0, 30)) for _ in range(k)]
        verificar(poli, 32, 32)
    print("Scanline confere com o teste par-impar em 202 poligonos.")

    # 2) Conferencia de area: U = 19*15 - 7*9 = 222 pixels
    grade, _ = scanline_fill(POLIGONO_U, 24, 20)
    print(f"Pixels do U: {int(grade.sum())} (area esperada 222)\n")

    # 3) Figura e exemplo numerico de uma scanline
    inter, y = figura(segs, raios, os.path.join(saida, "scanline_fill.png"))
    print(f"Scanline y = {y}: cruzamentos x = {inter[y]}")
    pares = list(zip(inter[y][0::2], inter[y][1::2]))
    print(f"Pares preenchidos: {pares}")
    # 4) Poligono dos extremos da arvore
    print()
    pontos, indices, raios2 = carregar_modelo_2d(sys.argv[1])
    figura_extremos(pontos, indices, raios2,
                    os.path.join(saida, "scanline_extremos.png"))
    print(f"\nFiguras salvas em {os.path.normpath(saida)}")