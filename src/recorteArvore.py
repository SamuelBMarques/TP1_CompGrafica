"""
recorteArvore.py

Modulo 6 do TP1 aplicado a arvore CCO fornecida pelo usuario: le o .vtk
informado, recorta TODOS os segmentos contra uma janela retangular com
Cohen-Sutherland e Liang-Barsky (recorteSegmentos.py) e gera as figuras
antes/depois.

Se nenhuma janela for informada, o programa procura sozinho, sobre a
caixa envolvente DA ARVORE LIDA, uma janela que corte o maior numero de
segmentos (e ainda deixe alguns inteiramente dentro).

Uso:
    python recorteArvore.py ../data/arvore.vtk
    python recorteArvore.py ../data/arvore.vtk --janela XMIN XMAX YMIN YMAX
    python recorteArvore.py ../data/arvore.vtk --tamanho 0.4 --saida minhas_figuras
Gera recorte_arvore.png (grade de pixels, como o Modulo 3) e
recorte_exemplo.png na pasta de saida
(padrao: ../report/figures).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from bresenham import bresenham, desenhar_grade
from recorteSegmentos import (carregar_segmentos, classificar_cs,
                              cohen_sutherland, figura_arvore,
                              figura_exemplo, liang_barsky)
from util import criar_mapeamento

CORES_CLASSE = {"aceito (trivial)": "tab:green", "rejeitado (trivial)": "tab:red",
                "recortado": "tab:blue", "rejeitado (apos recortes)": "tab:purple"}
CINZA = np.array([0.82, 0.82, 0.82])


def escolher_janela(segs, fracao=0.5, passos=11):
    """
    Varre janelas de tamanho (fracao x fracao) da caixa envolvente e devolve a
    que gera mais segmentos recortados (desempate: mais segmentos aceitos).
    """
    pts = segs.reshape(-1, 2)
    lo, tam = pts.min(axis=0), pts.max(axis=0) - pts.min(axis=0)
    larg, alt = fracao * tam
    melhor, melhor_pont = None, -1.0
    for fx in np.linspace(0, 1 - fracao, passos):
        for fy in np.linspace(0, 1 - fracao, passos):
            j = (lo[0] + fx * tam[0], lo[0] + fx * tam[0] + larg,
                 lo[1] + fy * tam[1], lo[1] + fy * tam[1] + alt)
            cats = [c for c, _ in classificar_cs(segs, j)]
            pont = cats.count("recortado") + 0.5 * cats.count("aceito (trivial)")
            if pont > melhor_pont:
                melhor, melhor_pont = j, pont
    return melhor


def rasterizar_camada(img, segs, cores, mapear):
    """
    Desenha segmentos (coordenadas do mundo) na imagem n x n com Bresenham,
    exatamente como bresenham.rasterizar_arvore: mapeia para pixels, arredonda
    as extremidades e pinta cada pixel com a cor do segmento.
    """
    n = img.shape[0]
    for (p0, p1), cor in zip(segs, cores):
        x0, y0 = mapear(*p0)
        x1, y1 = mapear(*p1)
        for x, y in bresenham(round(x0), round(y0), round(x1), round(y1)):
            if 0 <= x < n and 0 <= y < n:
                img[y, x] = cor


def figura_arvore_pixels(segs, raios, janela, caminho_saida, n=64):
    """
    Antes/depois do recorte na grade de pixels (mesmo estilo do Modulo 3):
    o recorte e feito em coordenadas do mundo e o resultado e rasterizado com
    Bresenham, usando o mesmo mapeamento mundo -> pixel da arvore inteira.
    """
    mapear = criar_mapeamento(segs, n, n)
    cs = [cohen_sutherland(*p0, *p1, janela) for p0, p1 in segs]
    lb = [liang_barsky(*p0, *p1, janela) for p0, p1 in segs]
    classes = classificar_cs(segs, janela)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(raios.min(), raios.max())
    cores_raio = cmap(norm(raios))[:, :3]

    def vazia():
        return np.ones((n, n, 3))

    antes = vazia()
    rasterizar_camada(antes, segs, cores_raio, mapear)

    decisao = vazia()
    rasterizar_camada(decisao, segs, [to_rgb(CORES_CLASSE[c]) for c, _ in classes], mapear)

    def depois(res):
        img = vazia()
        rasterizar_camada(img, segs, [CINZA] * len(segs), mapear)   # parte descartada
        vis = [(r[:2], r[2:]) for r in res if r is not None]
        cor = [cores_raio[i] for i, r in enumerate(res) if r is not None]
        rasterizar_camada(img, vis, cor, mapear)                    # parte visivel
        return img, len(vis)

    dep_cs, n_cs = depois(cs)
    dep_lb, n_lb = depois(lb)
    pix_dif = int(np.any(dep_cs != dep_lb, axis=2).sum())

    # janela em pixels (mesmo mapeamento da arvore)
    wx0, wy0 = mapear(janela[0], janela[2])
    wx1, wy1 = mapear(janela[1], janela[3])

    fig, eixos = plt.subplots(2, 2, figsize=(12, 12))
    paineis = [(eixos[0, 0], antes, f"Antes do recorte ({len(segs)} segmentos, {n}x{n} pixels)"),
               (eixos[0, 1], decisao, "Decisao do Cohen-Sutherland"),
               (eixos[1, 0], dep_cs, f"Depois: Cohen-Sutherland ({n_cs} trechos visiveis)"),
               (eixos[1, 1], dep_lb, f"Depois: Liang-Barsky ({n_lb} trechos visiveis)")]
    for ax, img, titulo in paineis:
        ax.imshow(img, origin="lower", interpolation="nearest")
        desenhar_grade(ax, n, n)
        ax.add_patch(plt.Rectangle((wx0, wy0), wx1 - wx0, wy1 - wy0, fill=False,
                                   edgecolor="red", linewidth=1.8))
        ax.set_title(titulo)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=eixos[0, 0], shrink=0.7, label="raio")
    contagem = {c: sum(1 for k, _ in classes if k == c) for c in CORES_CLASSE}
    eixos[0, 1].legend(handles=[plt.Line2D([], [], color=cor, linewidth=3,
                                           label=f"{c}: {contagem[c]}")
                                for c, cor in CORES_CLASSE.items()],
                       loc="upper left", fontsize=8)
    eixos[1, 1].text(0.02, 0.02, f"pixels diferentes entre CS e LB: {pix_dif}",
                     transform=eixos[1, 1].transAxes, fontsize=8,
                     bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return classes, pix_dif, (int(np.any(antes != 1, axis=2).sum()),
                              int((np.any(dep_cs != 1, axis=2) & np.any(dep_cs != CINZA, axis=2)).sum()))


def tabela(segs, raios, janela):
    """Imprime, para cada segmento, a decisao e o trecho que sobrou."""
    classes = classificar_cs(segs, janela)
    print(" seg   raio       decisao                    trecho visivel")
    for i, ((p0, p1), r, (cat, _)) in enumerate(zip(segs, raios, classes)):
        res = cohen_sutherland(*p0, *p1, janela)
        trecho = ("-" if res is None else
                  f"({res[0]:+.4f},{res[1]:+.4f}) -> ({res[2]:+.4f},{res[3]:+.4f})")
        print(f" {i:3d}  {r:.4f}  {cat:26s} {trecho}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arquivo", help="arquivo .vtk da arvore")
    ap.add_argument("--janela", type=float, nargs=4,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="janela de recorte em coordenadas do mundo")
    ap.add_argument("--tamanho", type=float, default=0.5,
                    help="lado da janela automatica, como fracao da caixa "
                         "envolvente (padrao 0.5)")
    ap.add_argument("--saida", help="pasta de saida das figuras")
    ap.add_argument("--n", type=int, default=64,
                    help="lado da grade de pixels da figura (padrao 64, como nos demais modulos)")
    ap.add_argument("--vetorial", action="store_true",
                    help="salva tambem a versao vetorial (recorte_arvore_vetorial.png)")
    a = ap.parse_args()

    segs, raios = carregar_segmentos(a.arquivo)
    pts = segs.reshape(-1, 2)
    print(f"Arquivo: {a.arquivo}")
    print(f"  {len(segs)} segmentos | raio [{raios.min():.4f}, {raios.max():.4f}]")
    print(f"  x em [{pts[:, 0].min():.4f}, {pts[:, 0].max():.4f}] | "
          f"y em [{pts[:, 1].min():.4f}, {pts[:, 1].max():.4f}]")

    janela = tuple(a.janela) if a.janela else escolher_janela(segs, a.tamanho)
    origem = "informada" if a.janela else "escolhida automaticamente"
    print("Janela %s: x in [%.4f, %.4f], y in [%.4f, %.4f]\n" % ((origem,) + janela))

    saida = a.saida or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "report", "figures")
    os.makedirs(saida, exist_ok=True)

    if len(segs) <= 40:
        tabela(segs, raios, janela)
        print()

    classes, pix_dif, (pix_antes, pix_depois) = figura_arvore_pixels(
        segs, raios, janela, os.path.join(saida, "recorte_arvore.png"), a.n)
    if a.vetorial:
        figura_arvore(segs, raios, janela,
                      os.path.join(saida, "recorte_arvore_vetorial.png"))
    figura_exemplo(os.path.join(saida, "recorte_exemplo.png"))
    for cat in ("aceito (trivial)", "rejeitado (trivial)", "recortado",
                "rejeitado (apos recortes)"):
        print(f"{cat:26s}: {sum(1 for c, _ in classes if c == cat)}")
    print(f"Pixels pintados na grade {a.n}x{a.n}: antes = {pix_antes}, depois = {pix_depois}")
    print(f"Pixels diferentes entre Cohen-Sutherland e Liang-Barsky: {pix_dif}")
    print(f"Figuras salvas em {os.path.normpath(saida)}")