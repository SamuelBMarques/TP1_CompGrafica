"""
scanline.py

Modulo 4 do TP1: preenchimento de poligonos com o algoritmo Scanline Fill,
implementado pelo grupo. O Matplotlib e usado apenas para mostrar a grade.

Convencao: o centro do pixel (i, j) fica na coordenada inteira (i, j).
Um pixel e pintado se o seu centro esta dentro do poligono.

Uso:
    python3 src/scanlineFill.py ../GRUPO_/data/tree2D_Nterm008.vtk
Gera ../report/figures/scanline_fill.png
"""

import os
import random
import sys
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from bresenham import desenhar_grade, rasterizar_arvore
from util import carregar_segmentos_2d


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
    print(f"\nFigura salva em {os.path.normpath(saida)}")