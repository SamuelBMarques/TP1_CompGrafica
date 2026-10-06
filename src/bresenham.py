"""
bresenham.py

Modulo 3 do TP1: rasterizacao de segmentos de reta com o algoritmo de
Bresenham, implementado pelo grupo (sem usar funcoes de desenho de linha
de nenhuma biblioteca). O Matplotlib e usado apenas para MOSTRAR a grade.

Uso:
    python3 src/bresenham.py ../GRUPO_/data/tree2D_Nterm008.vtk
Gera as figuras bresenham_exemplo.png e bresenham_arvore.png em ../report/figures
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from util import carregar_segmentos_2d, criar_mapeamento


# ---------------------------------------------------------------------------
# Algoritmo
# ---------------------------------------------------------------------------

def bresenham(x0, y0, x1, y1, rastro=None):
    """
    Devolve a lista de pixels (x, y) do segmento de (x0, y0) a (x1, y1).
    Todas as coordenadas sao inteiras e so ha somas e comparacoes de inteiros.

    Funciona em todos os octantes. A ideia:
      - dx e dy sao os deslocamentos absolutos em x e y;
      - 'erro' acumula a diferenca entre a reta ideal e o pixel escolhido;
      - a cada passo, se o erro indica que ficamos longe demais da reta
        em x, andamos em x; se ficamos longe em y, andamos em y;
      - quando os dois ajustes ocorrem, o passo e diagonal.
    Se 'rastro' for uma lista, guarda (x, y, erro) de cada passo (didatico).
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1       # sentido de avanco em x
    sy = 1 if y0 < y1 else -1       # sentido de avanco em y

    erro = dx - dy
    x, y = x0, y0
    pixels = []

    while True:
        pixels.append((x, y))
        if rastro is not None:
            rastro.append((x, y, erro))
        if x == x1 and y == y1:
            break

        e2 = 2 * erro               # dobro do erro evita usar fracoes
        if e2 > -dy:                # precisa avancar em x
            erro -= dy
            x += sx
        if e2 < dx:                 # precisa avancar em y
            erro += dx
            y += sy

    return pixels


# ---------------------------------------------------------------------------
# Verificacao automatica
# ---------------------------------------------------------------------------

def verificar(x0, y0, x1, y1):
    """Confere propriedades que todo Bresenham correto deve ter."""
    px = bresenham(x0, y0, x1, y1)
    dx, dy = abs(x1 - x0), abs(y1 - y0)

    assert px[0] == (x0, y0) and px[-1] == (x1, y1), "extremos"
    assert len(px) == max(dx, dy) + 1, "quantidade de pixels"
    for (a, b), (c, d) in zip(px[:-1], px[1:]):
        assert max(abs(c - a), abs(d - b)) == 1, "pixels vizinhos"

    # Cada pixel deve ficar a no maximo 0,5 da reta ideal no eixo menor
    for x, y in px:
        if dx >= dy:
            ideal = y0 + (y1 - y0) * (x - x0) / (x1 - x0) if dx else y0
            assert abs(y - ideal) <= 0.5 + 1e-9, "desvio da reta ideal"
        else:
            ideal = x0 + (x1 - x0) * (y - y0) / (y1 - y0)
            assert abs(x - ideal) <= 0.5 + 1e-9, "desvio da reta ideal"


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def desenhar_grade(ax, largura, altura):
    """Configura os eixos para mostrar uma grade de pixels."""
    ax.set_xlim(-0.5, largura - 0.5)
    ax.set_ylim(-0.5, altura - 0.5)
    ax.set_xticks(np.arange(-0.5, largura, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, altura, 1), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_aspect("equal")


def figura_exemplo(caminho_saida):
    """Exemplo numerico: segmento (0,0) -> (8,3)."""
    rastro = []
    px = bresenham(0, 0, 8, 3, rastro)

    print("Exemplo: (0,0) -> (8,3)")
    print(" passo   pixel    erro (antes da decisao)")
    for i, (x, y, e) in enumerate(rastro):
        print(f" {i:4d}   ({x},{y})   {e:4d}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    desenhar_grade(ax, 10, 5)
    ax.set_xticks(range(10))
    ax.set_yticks(range(5))
    for x, y in px:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="tab:blue", alpha=0.6))
    ax.plot([0, 8], [0, 3], "r-", linewidth=1.5, label="reta ideal")
    ax.plot(*zip(*px), "k.", markersize=4, label="centro dos pixels")
    ax.set_title("Bresenham: (0,0) -> (8,3)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def rasterizar_arvore(segs, raios, n):
    """Rasteriza todos os vasos numa grade n x n. Cada pixel guarda o raio."""
    mapear = criar_mapeamento(segs, n, n, margem=2)
    grade = np.full((n, n), np.nan)
    for (p0, p1), r in zip(segs, raios):
        x0, y0 = mapear(*p0)
        x1, y1 = mapear(*p1)
        # Bresenham trabalha com inteiros: arredonda as extremidades
        for x, y in bresenham(round(x0), round(y0), round(x1), round(y1)):
            grade[y, x] = r
    return grade, mapear


def figura_arvore(segs, raios, caminho_saida, n=64, seg_zoom=3):
    grade, mapear = rasterizar_arvore(segs, raios, n)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))

    # Painel 1: arvore inteira na grade
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    img = a1.imshow(grade, origin="lower", cmap=cmap,
                    vmin=raios.min(), vmax=raios.max(), interpolation="nearest")
    desenhar_grade(a1, n, n)
    a1.set_title(f"Arvore CCO rasterizada ({n}x{n} pixels)")
    fig.colorbar(img, ax=a1, shrink=0.7, label="raio")

    # Painel 2: zoom em um vaso, comparando pixels e reta ideal
    (p0, p1) = segs[seg_zoom]
    x0, y0 = mapear(*p0)
    x1, y1 = mapear(*p1)
    xi0, yi0, xi1, yi1 = round(x0), round(y0), round(x1), round(y1)
    px = bresenham(xi0, yi0, xi1, yi1)

    xs, ys = zip(*px)
    xmin, xmax = min(xs) - 2, max(xs) + 3
    ymin, ymax = min(ys) - 2, max(ys) + 3
    for x, y in px:
        a2.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="tab:blue", alpha=0.6))
    a2.plot([xi0, xi1], [yi0, yi1], "r-", linewidth=1.2, label="reta ideal")
    a2.plot(xs, ys, "k.", markersize=3, label="centro dos pixels")
    a2.set_xlim(xmin - 0.5, xmax - 0.5)
    a2.set_ylim(ymin - 0.5, ymax - 0.5)
    a2.set_xticks(np.arange(xmin - 0.5, xmax, 1), minor=True)
    a2.set_yticks(np.arange(ymin - 0.5, ymax, 1), minor=True)
    a2.grid(which="minor", color="0.85", linewidth=0.5)
    a2.tick_params(which="minor", length=0)
    a2.set_aspect("equal")
    a2.set_title(f"Zoom no vaso {seg_zoom}: ({xi0},{yi0}) -> ({xi1},{yi1}), {len(px)} pixels")
    a2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python bresenham.py arquivo.vtk")
        sys.exit(1)

    segs, raios = carregar_segmentos_2d(sys.argv[1])
    saida = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "report", "figures")
    os.makedirs(saida, exist_ok=True)

    # 1) Testes automaticos em segmentos de todas as direcoes
    for args in [(0, 0, 8, 3), (8, 3, 0, 0), (0, 0, 3, 8), (0, 8, 5, 0),
                 (5, 5, 5, 5), (0, 0, 6, 0), (0, 0, 0, 6), (7, 2, 1, 5)]:
        verificar(*args)
    mapear = criar_mapeamento(segs, 64, 64)
    for p0, p1 in segs:
        a, b = mapear(*p0), mapear(*p1)
        verificar(round(a[0]), round(a[1]), round(b[0]), round(b[1]))
    print("Todos os testes de Bresenham passaram.\n")

    # 2) Figuras
    figura_exemplo(os.path.join(saida, "bresenham_exemplo.png"))
    figura_arvore(segs, raios, os.path.join(saida, "bresenham_arvore.png"))
    print(f"\nFiguras salvas em {os.path.normpath(saida)}")