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
Gera recorte_arvore.png (e recorte_exemplo.png) na pasta de saida
(padrao: ../report/figures).
"""

import argparse
import os

import numpy as np

from recorteSegmentos import (carregar_segmentos, classificar_cs,
                              cohen_sutherland, figura_arvore,
                              figura_exemplo, liang_barsky)


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

    classes, dif = figura_arvore(segs, raios, janela,
                                 os.path.join(saida, "recorte_arvore.png"))
    figura_exemplo(os.path.join(saida, "recorte_exemplo.png"))
    for cat in ("aceito (trivial)", "rejeitado (trivial)", "recortado",
                "rejeitado (apos recortes)"):
        print(f"{cat:26s}: {sum(1 for c, _ in classes if c == cat)}")
    print(f"Diferenca maxima Cohen-Sutherland x Liang-Barsky: {dif:.2e}")
    print(f"Figuras salvas em {os.path.normpath(saida)}")
