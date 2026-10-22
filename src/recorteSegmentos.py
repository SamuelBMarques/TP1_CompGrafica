"""
recorteSegmentos.py

Modulo 6 do TP1: recorte de segmentos de reta contra uma janela retangular
alinhada aos eixos, implementado pelo grupo (sem funcoes de recorte de
nenhuma biblioteca). O Matplotlib e usado apenas para MOSTRAR os resultados.

Algoritmos:
  1. Cohen-Sutherland (implementacao obrigatoria)
  2. Liang-Barsky    (o escolhido entre Liang-Barsky e Cyrus-Beck)
  3. Cyrus-Beck      (somente discutido no relatorio; ver o fim deste arquivo)

Convencao: a janela e a tupla (xmin, xmax, ymin, ymax) e a borda pertence
a janela (um ponto sobre a borda esta "dentro"). Todos os calculos sao em
ponto flutuante, no mesmo espaco em que os segmentos estao definidos
(aqui, coordenadas do mundo; poderiam ser pixels).

Uso:
    python recorteSegmentos.py ../data/exemplo2d.vtk
    python recorteSegmentos.py ../data/exemplo2d.vtk --janela 0.02 0.08 0.01 0.07
Gera ../report/figures/recorte_exemplo.png e recorte_arvore.png
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection



def carregar_segmentos(caminho):
    """
    Le o VTK e devolve (segs (M,2,2), raios (M,)), usando so x e y.
    Usa o util.py (PyVista); se o PyVista nao estiver instalado, cai no
    leitor manual readVtk.py (VTK ASCII), assim o modulo roda de qualquer forma.
    """
    try:
        from util import carregar_segmentos_2d
        return carregar_segmentos_2d(caminho)
    except ImportError:
        from readVtk import ler_vtk
        m = ler_vtk(caminho)
        raios = m.raios if m.raios is not None else np.ones(m.num_segmentos)
        return m.pontos[m.segmentos][:, :, :2], np.asarray(raios, dtype=float)

# ---------------------------------------------------------------------------
# Cohen-Sutherland
# ---------------------------------------------------------------------------

# Codigo de regiao de 4 bits: [ACIMA ABAIXO DIREITA ESQUERDA]
DENTRO, ESQUERDA, DIREITA, ABAIXO, ACIMA = 0, 1, 2, 4, 8


def codigo_regiao(x, y, janela):
    """Outcode do ponto (x, y): cada bit ligado diz de que lado da janela ele esta."""
    xmin, xmax, ymin, ymax = janela
    c = DENTRO
    if x < xmin:
        c |= ESQUERDA
    elif x > xmax:
        c |= DIREITA
    if y < ymin:
        c |= ABAIXO
    elif y > ymax:
        c |= ACIMA
    return c


def cohen_sutherland(x0, y0, x1, y1, janela, rastro=None):
    """
    Recorta o segmento (x0,y0)-(x1,y1) contra a janela.
    Devolve (x0', y0', x1', y1') do trecho visivel ou None se nada aparece.
    Se 'rastro' for uma lista, guarda (x0, y0, c0, x1, y1, c1) a cada
    iteracao (didatico).

    Ideia:
      - aceitacao trivial: os dois codigos sao 0 (c0 | c1 == 0);
      - rejeicao trivial : os dois pontos estao do mesmo lado de fora
                           (c0 & c1 != 0);
      - caso contrario, escolhe um extremo que esteja fora, move-o ate a
        borda indicada por um de seus bits e repete com o segmento menor.
    """
    xmin, xmax, ymin, ymax = janela
    c0 = codigo_regiao(x0, y0, janela)
    c1 = codigo_regiao(x1, y1, janela)

    # Cada iteracao zera pelo menos um bit; sao no maximo 4 recortes. O limite
    # de 10 so protege contra um laco infinito causado por arredondamento.
    for _ in range(10):
        if rastro is not None:
            rastro.append((x0, y0, c0, x1, y1, c1))
        if not (c0 | c1):
            return (x0, y0, x1, y1)                  # aceitacao trivial
        if c0 & c1:
            return None                              # rejeicao trivial

        c_fora = c0 if c0 else c1                    # extremo que sera movido
        # Intersecao com a reta da borda. As divisoes sao seguras: se o
        # extremo esta acima de ymax e o outro nao (c0 & c1 == 0), entao
        # y1 != y0 (idem para as demais bordas).
        if c_fora & ACIMA:
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
            y = ymax
        elif c_fora & ABAIXO:
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
            y = ymin
        elif c_fora & DIREITA:
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
            x = xmax
        else:  # ESQUERDA
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
            x = xmin

        if c_fora == c0:
            x0, y0 = x, y
            c0 = codigo_regiao(x0, y0, janela)
        else:
            x1, y1 = x, y
            c1 = codigo_regiao(x1, y1, janela)
    return None


# ---------------------------------------------------------------------------
# Liang-Barsky
# ---------------------------------------------------------------------------

def liang_barsky(x0, y0, x1, y1, janela, rastro=None):
    """
    Recorte parametrico. O segmento e P(t) = P0 + t*(P1 - P0), t em [0, 1].
    Para cada borda k vale  p_k * t <= q_k  para P(t) estar do lado de dentro:

        borda       p_k    q_k
        esquerda   -dx    x0 - xmin
        direita    +dx    xmax - x0
        abaixo     -dy    y0 - ymin
        acima      +dy    ymax - y0

      p_k == 0 : segmento paralelo a borda; se q_k < 0 esta todo fora.
      p_k <  0 : o segmento ENTRA pela borda em t = q_k/p_k  -> t0 = max(t0, t)
      p_k >  0 : o segmento SAI   pela borda em t = q_k/p_k  -> t1 = min(t1, t)
    Se em algum momento t0 > t1, nao ha trecho visivel.
    Devolve (x0', y0', x1', y1') ou None. 'rastro' guarda
    (borda, p, q, t, efeito) de cada borda (didatico).
    """
    xmin, xmax, ymin, ymax = janela
    dx, dy = x1 - x0, y1 - y0
    bordas = (("esquerda", -dx, x0 - xmin), ("direita", dx, xmax - x0),
              ("abaixo", -dy, y0 - ymin), ("acima", dy, ymax - y0))

    t0, t1 = 0.0, 1.0
    for nome, p, q in bordas:
        if p == 0:                                   # paralelo a borda
            efeito = "paralelo, fora: rejeita" if q < 0 else "paralelo, dentro"
            if rastro is not None:
                rastro.append((nome, p, q, None, efeito))
            if q < 0:
                return None
            continue

        t = q / p
        if p < 0:                                    # entrando
            t0 = max(t0, t)
            efeito = f"entra: t0 = {t0:.4f}"
        else:                                        # saindo
            t1 = min(t1, t)
            efeito = f"sai: t1 = {t1:.4f}"
        if rastro is not None:
            rastro.append((nome, p, q, t, efeito))
        if t0 > t1:
            return None

    return (x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy)


# ---------------------------------------------------------------------------
# Verificacao automatica
# ---------------------------------------------------------------------------

def verificar_recorte(seg, janela, cs, lb, tol=1e-9):
    """
    Confere, para um segmento, propriedades que todo recorte correto tem:
      - os dois algoritmos concordam (visibilidade e extremos);
      - extremos recortados estao na janela e sobre a reta original;
      - o trecho e MAXIMO: um passo alem de cada extremo recortado, em
        direcao ao extremo original, cai fora da janela;
      - se foi rejeitado, nenhum ponto amostrado do segmento esta dentro.
    """
    (x0, y0), (x1, y1) = seg
    xmin, xmax, ymin, ymax = janela
    assert (cs is None) == (lb is None), f"visibilidade difere em {seg}"

    if cs is None:
        for t in np.linspace(0.0, 1.0, 1001):
            ponto = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            assert codigo_regiao(*ponto, janela) != 0, f"rejeitado a torto {seg}"
        return

    assert np.allclose(cs, lb, atol=tol), f"extremos diferem em {seg}: {cs} x {lb}"

    larg = (xmin - tol, xmax + tol, ymin - tol, ymax + tol)
    d = np.array([x1 - x0, y1 - y0])
    comp = np.hypot(*d)
    u = d / comp if comp > 0 else d
    a, b = np.array(cs[:2]), np.array(cs[2:])
    for p in (a, b):
        assert codigo_regiao(*p, larg) == 0, f"extremo fora da janela em {seg}"
        cruz = (p[0] - x0) * d[1] - (p[1] - y0) * d[0]
        assert abs(cruz) <= 1e-9 * max(1.0, comp * comp), f"fora da reta em {seg}"
    assert np.dot(b - a, d) >= -tol, f"orientacao invertida em {seg}"

    passo = 1e-5 * max(1.0, comp)
    if np.hypot(a[0] - x0, a[1] - y0) > 1e-7 and comp > 0:
        p = a - u * passo
        assert codigo_regiao(*p, janela) != 0, f"inicio nao maximal em {seg}"
    if np.hypot(b[0] - x1, b[1] - y1) > 1e-7 and comp > 0:
        p = b + u * passo
        assert codigo_regiao(*p, janela) != 0, f"fim nao maximal em {seg}"


def testar_casos_fixos():
    """Cenario de teste: um segmento para cada situacao tipica."""
    janela = (2, 8, 2, 6)
    casos = [  # (nome, p0, p1, visivel?)
        ("totalmente dentro",            (3, 3), (7, 5), True),
        ("fora, mesmo lado",             (0, 0), (1, 7), False),
        ("atravessa a janela",           (0, 4), (10, 4), True),
        ("fora, rejeitado so apos recorte", (0, 5), (3, 8), False),
        ("um extremo dentro",            (5, 4), (12, 4), True),
        ("vertical atravessando",        (5, 0), (5, 9), True),
        ("horizontal fora",              (0, 7), (10, 7), False),
        ("ponto dentro",                 (4, 4), (4, 4), True),
        ("ponto fora",                   (0, 0), (0, 0), False),
        ("sobre a borda",                (2, 0), (2, 9), True),
        ("diagonal 2 cantos",            (0, 0), (10, 8), True),
    ]
    for nome, p0, p1, esperado in casos:
        cs = cohen_sutherland(*p0, *p1, janela)
        lb = liang_barsky(*p0, *p1, janela)
        assert (cs is not None) == esperado, f"CS errou em '{nome}'"
        assert (lb is not None) == esperado, f"LB errou em '{nome}'"
        verificar_recorte((p0, p1), janela, cs, lb)
    return len(casos)


def testar_aleatorios(n=20000, semente=1):
    """Segmentos aleatorios (reais e em grade inteira, que gera casos degenerados)."""
    rng = np.random.default_rng(semente)
    janela = (0.0, 1.0, 0.0, 1.0)
    for p in rng.uniform(-1, 2, size=(n, 2, 2)):
        seg = (tuple(p[0]), tuple(p[1]))
        verificar_recorte(seg, janela, cohen_sutherland(*seg[0], *seg[1], janela),
                          liang_barsky(*seg[0], *seg[1], janela))
    janela = (2, 4, 2, 4)
    for p in rng.integers(0, 7, size=(n, 2, 2)):
        seg = ((int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])))
        verificar_recorte(seg, janela, cohen_sutherland(*seg[0], *seg[1], janela),
                          liang_barsky(*seg[0], *seg[1], janela))
    return 2 * n


# ---------------------------------------------------------------------------
# Comparacao entre os algoritmos
# ---------------------------------------------------------------------------

def classificar_cs(segs, janela):
    """Como o Cohen-Sutherland decidiu cada segmento: (categoria, no de recortes)."""
    resultado = []
    for p0, p1 in segs:
        rastro = []
        r = cohen_sutherland(*p0, *p1, janela, rastro)
        recortes = len(rastro) - 1                   # 1a entrada = estado inicial
        if recortes == 0:
            cat = "aceito (trivial)" if r else "rejeitado (trivial)"
        else:
            cat = "recortado" if r else "rejeitado (apos recortes)"
        resultado.append((cat, recortes))
    return resultado


def comparar_tempo(n=20000, repeticoes=3):
    """Tempo (Python puro, so como ordem de grandeza) nos mesmos segmentos."""
    rng = np.random.default_rng(0)
    janela = (0.0, 1.0, 0.0, 1.0)
    segs = [(tuple(p[0]), tuple(p[1])) for p in rng.uniform(-1, 2, size=(n, 2, 2))]
    tempos = {}
    for nome, f in (("Cohen-Sutherland", cohen_sutherland), ("Liang-Barsky", liang_barsky)):
        melhor = min(_cronometrar(f, segs, janela) for _ in range(repeticoes))
        tempos[nome] = melhor
    return tempos


def _cronometrar(f, segs, janela):
    t = time.perf_counter()
    for p0, p1 in segs:
        f(*p0, *p1, janela)
    return time.perf_counter() - t


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def figura_exemplo(caminho_saida):
    """Exemplo numerico passo a passo do Cohen-Sutherland (e do Liang-Barsky)."""
    janela = (2, 8, 2, 6)
    p0, p1 = (-1, 7), (10, 1)

    rastro = []
    res = cohen_sutherland(*p0, *p1, janela, rastro)
    print(f"Exemplo: janela x in [2, 8], y in [2, 6]; segmento {p0} -> {p1}")
    print("Cohen-Sutherland (codigo = ACIMA ABAIXO DIREITA ESQUERDA):")
    print(" iter    P0 (codigo)              P1 (codigo)")
    for i, (xa, ya, ca, xb, yb, cb) in enumerate(rastro):
        print(f" {i:3d}   ({xa:7.3f},{ya:7.3f}) {ca:04b}    ({xb:7.3f},{yb:7.3f}) {cb:04b}")
    print(f" resultado: {tuple(round(v, 3) for v in res)}\n")

    rl = []
    res_lb = liang_barsky(*p0, *p1, janela, rl)
    print("Liang-Barsky no mesmo segmento:")
    print(" borda      p        q        t       efeito")
    for nome, p, q, t, ef in rl:
        ts = "   -  " if t is None else f"{t:7.4f}"
        print(f" {nome:8s} {p:7.2f} {q:8.2f}  {ts}   {ef}")
    print(f" resultado: {tuple(round(v, 3) for v in res_lb)}\n")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    xmin, xmax, ymin, ymax = janela
    for v in (xmin, xmax):
        ax.axvline(v, color="0.6", linestyle=":", linewidth=1)
    for v in (ymin, ymax):
        ax.axhline(v, color="0.6", linestyle=":", linewidth=1)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=True, facecolor="tab:orange", alpha=0.15,
                               edgecolor="red", linewidth=1.8))
    # codigo de cada uma das 9 regioes
    for cx in (0.0, 5.0, 9.5):
        for cy in (0.5, 4.0, 7.25):
            ax.text(cx, cy, f"{codigo_regiao(cx, cy, janela):04b}", ha="center",
                    va="center", color="0.45", fontsize=11, family="monospace")
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "--", color="0.5", label="segmento original")
    ax.plot([res[0], res[2]], [res[1], res[3]], "-", color="tab:blue", linewidth=3.5,
            label="trecho visivel")
    ax.plot(*zip(p0, p1), "ks", markersize=6)
    ax.annotate("P0", p0, textcoords="offset points", xytext=(-14, 4))
    ax.annotate("P1", p1, textcoords="offset points", xytext=(6, 4))
    for i in range(1, len(rastro)):                  # pontos criados em cada recorte
        a, b = rastro[i - 1], rastro[i]
        novo = (b[0], b[1]) if (a[0], a[1]) != (b[0], b[1]) else (b[3], b[4])
        ax.plot(*novo, "o", color="tab:red", markersize=7)
        ax.annotate(str(i), novo, textcoords="offset points",
                    xytext=(6, -12) if i % 2 else (-14, 6),
                    color="tab:red", fontsize=10, weight="bold")
    ax.set_xlim(-2, 11)
    ax.set_ylim(-1, 8.5)
    ax.set_aspect("equal")
    ax.set_title("Cohen-Sutherland: codigos de regiao e recortes sucessivos")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def _colecao(segs, **kw):
    return LineCollection([[tuple(p0), tuple(p1)] for p0, p1 in segs], **kw)


def figura_arvore(segs, raios, janela, caminho_saida):
    """Antes/depois do recorte sobre a arvore CCO (2x2 paineis)."""
    cs = [cohen_sutherland(*p0, *p1, janela) for p0, p1 in segs]
    lb = [liang_barsky(*p0, *p1, janela) for p0, p1 in segs]
    dif = max((np.max(np.abs(np.array(a) - np.array(b)))
               for a, b in zip(cs, lb) if a is not None and b is not None), default=0.0)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(raios.min(), raios.max())
    cores_classe = {"aceito (trivial)": "tab:green", "rejeitado (trivial)": "tab:red",
                    "recortado": "tab:blue", "rejeitado (apos recortes)": "tab:purple"}
    classes = classificar_cs(segs, janela)

    pts = segs.reshape(-1, 2)
    margem = 0.04 * (pts.max(axis=0) - pts.min(axis=0))
    lo, hi = pts.min(axis=0) - margem, pts.max(axis=0) + margem
    xmin, xmax, ymin, ymax = janela

    fig, eixos = plt.subplots(2, 2, figsize=(12, 12))

    def moldura(ax, titulo):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=True, facecolor="tab:orange", alpha=0.12,
                                   edgecolor="red", linewidth=1.8, zorder=1))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_title(titulo)

    # (a) antes
    ax = eixos[0, 0]
    ax.add_collection(_colecao(segs, colors=cmap(norm(raios)), linewidths=2, zorder=2))
    moldura(ax, f"Antes do recorte ({len(segs)} segmentos)")

    # (b) classificacao do Cohen-Sutherland
    ax = eixos[0, 1]
    ax.add_collection(_colecao(segs, colors=[cores_classe[c] for c, _ in classes],
                               linewidths=2, zorder=2))
    moldura(ax, "Decisao do Cohen-Sutherland")
    contagem = {c: sum(1 for k, _ in classes if k == c) for c in cores_classe}
    ax.legend(handles=[plt.Line2D([], [], color=cor, linewidth=2,
                                  label=f"{c}: {contagem[c]}")
                       for c, cor in cores_classe.items()], loc="upper left", fontsize=8)

    # (c) e (d) depois, para cada algoritmo
    for ax, res, nome in ((eixos[1, 0], cs, "Cohen-Sutherland"),
                          (eixos[1, 1], lb, "Liang-Barsky")):
        vis = [(r[:2], r[2:]) for r in res if r is not None]
        cor = [cmap(norm(raios[i])) for i, r in enumerate(res) if r is not None]
        ax.add_collection(_colecao(segs, colors="0.85", linewidths=1, zorder=2))
        ax.add_collection(_colecao(vis, colors=cor, linewidths=2.5, zorder=3))
        moldura(ax, f"Depois: {nome} ({len(vis)} trechos visiveis)")
    eixos[1, 1].text(0.02, 0.02, f"diferenca maxima p/ Cohen-Sutherland: {dif:.1e}",
                     transform=eixos[1, 1].transAxes, fontsize=8)

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return classes, dif


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arquivo", help="arquivo .vtk")
    ap.add_argument("--janela", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="janela de recorte em coordenadas do mundo "
                         "(padrao: regiao central da caixa envolvente)")
    a = ap.parse_args()

    segs, raios = carregar_segmentos(a.arquivo)
    saida = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "report", "figures")
    os.makedirs(saida, exist_ok=True)

    if a.janela:
        janela = tuple(a.janela)
    else:
        pts = segs.reshape(-1, 2)
        lo, tam = pts.min(axis=0), pts.max(axis=0) - pts.min(axis=0)
        janela = (lo[0] + 0.25 * tam[0], lo[0] + 0.70 * tam[0],
                  lo[1] + 0.30 * tam[1], lo[1] + 0.75 * tam[1])

    # 1) Testes automaticos
    n1 = testar_casos_fixos()
    n2 = testar_aleatorios()
    print(f"Recorte: {n1} casos fixos e {n2} segmentos aleatorios conferidos "
          "(CS = LB, extremos na janela, trecho maximal).\n")

    # 2) Exemplo numerico e figura didatica
    figura_exemplo(os.path.join(saida, "recorte_exemplo.png"))

    # 3) Arvore CCO antes/depois
    print("Janela da arvore: x in [%.4f, %.4f], y in [%.4f, %.4f]" % janela)
    classes, dif = figura_arvore(segs, raios, janela,
                                 os.path.join(saida, "recorte_arvore.png"))
    for cat in ("aceito (trivial)", "rejeitado (trivial)", "recortado",
                "rejeitado (apos recortes)"):
        print(f"  {cat:26s}: {sum(1 for c, _ in classes if c == cat)}")
    print(f"  diferenca maxima entre CS e LB: {dif:.2e}\n")

    # 4) Comparacao de custo
    tempos = comparar_tempo()
    print("Tempo para 20000 segmentos aleatorios (Python puro, melhor de 3):")
    for nome, t in tempos.items():
        print(f"  {nome:17s}: {t * 1000:7.1f} ms")
    print(f"\nFiguras salvas em {os.path.normpath(saida)}")


# ---------------------------------------------------------------------------
# Discussao (para o relatorio): Cyrus-Beck
# ---------------------------------------------------------------------------
# Cyrus-Beck e a base geral de Liang-Barsky. Ele recorta contra QUALQUER
# janela CONVEXA (poligono convexo; em 3D, poliedro convexo). Cada aresta tem
# uma normal N_i apontando para fora e um ponto P_i sobre ela. Com
# P(t) = P0 + t*d, o cruzamento com a aresta i ocorre em
#     t = N_i . (P_i - P0) / (N_i . d)
# e o sinal de N_i . d diz se o segmento ENTRA (< 0) ou SAI (> 0) da janela;
# t0 = max das entradas e t1 = min das saidas, como em Liang-Barsky.
# Liang-Barsky e o caso particular da janela retangular alinhada aos eixos:
# as normais sao (+-1, 0) e (0, +-1), os produtos escalares viram os p_k e q_k
# da tabela acima e nao sobra multiplicacao. A diferenca em relacao ao
# Cohen-Sutherland e que os dois recortam de forma parametrica (calculam t
# uma vez por borda e so no fim obtem os pontos), enquanto o Cohen-Sutherland
# calcula pontos de intersecao a cada iteracao, apoiado em codigos de regiao.
