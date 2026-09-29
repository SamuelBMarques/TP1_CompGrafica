"""
readVtk.py

Leitor simples de arquivos VTK legado (ASCII, DATASET POLYDATA) para
arvores arteriais CCO. Nao depende de VTK/PyVista: o arquivo e lido
"na mao", o que facilita explicar o formato no relatorio.

Uso pela linha de comando:
    python readVtk.py caminho/do/arquivo.vtk
"""

import sys
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModeloCCO:
    """Guarda os dados lidos do arquivo VTK."""
    pontos: np.ndarray                 # matriz (N, 3): coordenadas x, y, z
    segmentos: np.ndarray              # matriz (M, 2): indices dos dois pontos de cada vaso
    raios: np.ndarray | None = None    # vetor (M,): raio de cada segmento
    atributos_celula: dict = field(default_factory=dict)  # todos os atributos por segmento
    atributos_ponto: dict = field(default_factory=dict)   # todos os atributos por ponto

    @property
    def num_pontos(self):
        return len(self.pontos)

    @property
    def num_segmentos(self):
        return len(self.segmentos)

    @property
    def eh_2d(self):
        # Se todos os valores de z forem iguais a zero, o modelo e plano
        return bool(np.allclose(self.pontos[:, 2], 0.0))


def ler_vtk(caminho):
    """Le um arquivo VTK legado ASCII e devolve um objeto ModeloCCO."""

    with open(caminho, "r") as f:
        texto = f.read()

    # Cabecalho: linha 0 = versao, linha 1 = titulo, linha 2 = ASCII ou BINARY
    linhas = texto.splitlines()
    if not linhas[0].startswith("# vtk DataFile"):
        raise ValueError("Arquivo nao parece ser um VTK legado.")
    if linhas[2].strip().upper() != "ASCII":
        raise ValueError("Somente arquivos VTK em formato ASCII sao suportados.")

    # Do resto do arquivo em diante, separamos tudo em "palavras" (tokens).
    # Assim nao importa como os numeros estao distribuidos nas linhas.
    tokens = "\n".join(linhas[3:]).split()

    pontos = None
    segmentos = []
    atributos_celula = {}
    atributos_ponto = {}
    secao = None   # indica se estamos em CELL_DATA ou POINT_DATA
    n_dados = 0    # quantidade de valores declarada na secao atual

    i = 0
    while i < len(tokens):
        chave = tokens[i].upper()

        if chave == "POINTS":
            # Formato: POINTS <quantidade> <tipo>, seguido de 3 valores por ponto
            n = int(tokens[i + 1])
            i += 3
            valores = np.array(tokens[i:i + 3 * n], dtype=float)
            pontos = valores.reshape(n, 3)
            i += 3 * n

        elif chave == "LINES":
            # Formato: LINES <num_celulas> <total_de_inteiros>
            # Cada celula: <k> <id_1> ... <id_k>
            n_celulas = int(tokens[i + 1])
            total = int(tokens[i + 2])
            i += 3
            inteiros = [int(t) for t in tokens[i:i + total]]
            i += total

            p = 0
            for _ in range(n_celulas):
                k = inteiros[p]
                ids = inteiros[p + 1:p + 1 + k]
                p += 1 + k
                # Uma celula com k pontos vira k-1 segmentos consecutivos
                # (para k = 2, e simplesmente um segmento)
                for a, b in zip(ids[:-1], ids[1:]):
                    segmentos.append((a, b))

        elif chave in ("CELL_DATA", "POINT_DATA"):
            secao = chave
            n_dados = int(tokens[i + 1])
            i += 2

        elif chave == "SCALARS":
            # Formato: SCALARS <nome> <tipo> [num_componentes]
            #          LOOKUP_TABLE <nome_tabela>
            #          <valores>
            nome = tokens[i + 1]
            i += 3  # pula SCALARS, nome e tipo

            # O numero de componentes e opcional (padrao = 1)
            num_comp = 1
            if tokens[i].upper() != "LOOKUP_TABLE":
                num_comp = int(tokens[i])
                i += 1

            i += 2  # pula LOOKUP_TABLE e o nome da tabela (ignorada)

            qtd = n_dados * num_comp
            valores = np.array(tokens[i:i + qtd], dtype=float)
            i += qtd

            if secao == "CELL_DATA":
                atributos_celula[nome] = valores
            else:
                atributos_ponto[nome] = valores

        else:
            # Palavra desconhecida: apenas avanca
            i += 1

    if pontos is None or len(segmentos) == 0:
        raise ValueError("Arquivo sem POINTS ou sem LINES.")

    segmentos = np.array(segmentos, dtype=int)

    # Verifica se todos os indices apontam para pontos existentes
    if segmentos.min() < 0 or segmentos.max() >= len(pontos):
        raise ValueError("LINES referencia um ponto que nao existe.")

    # Escolhe o atributo de raio: prefere o nome "raio" (ou "radius")
    raios = None
    for nome, valores in atributos_celula.items():
        if nome.lower() in ("raio", "radius"):
            raios = valores
    if raios is None and atributos_celula:
        raios = next(iter(atributos_celula.values()))

    return ModeloCCO(pontos, segmentos, raios, atributos_celula, atributos_ponto)


def imprimir_resumo(modelo):
    """Mostra as informacoes pedidas no Modulo 1."""
    print("=== Resumo do modelo CCO ===")
    print(f"Dimensao          : {'2D' if modelo.eh_2d else '3D'}")
    print(f"Numero de pontos  : {modelo.num_pontos}")
    print(f"Numero de segmentos: {modelo.num_segmentos}")

    if modelo.raios is not None:
        r = modelo.raios
        print(f"Raio minimo       : {r.min():.7f} (segmento {r.argmin()})")
        print(f"Raio maximo       : {r.max():.7f} (segmento {r.argmax()})")

    # Caixa envolvente: util depois para converter mundo -> pixels
    minimo = modelo.pontos.min(axis=0)
    maximo = modelo.pontos.max(axis=0)
    print(f"Limites em x      : [{minimo[0]:.4f}, {maximo[0]:.4f}]")
    print(f"Limites em y      : [{minimo[1]:.4f}, {maximo[1]:.4f}]")
    print(f"Limites em z      : [{minimo[2]:.4f}, {maximo[2]:.4f}]")

    # Folhas: pontos que aparecem em apenas um segmento (exceto a raiz)
    grau = np.bincount(modelo.segmentos.ravel(), minlength=modelo.num_pontos)
    print(f"Pontos folha      : {int(np.sum(grau == 1)) - 1} (sem contar a raiz)")


def imprimir_segmentos(modelo):
    """Lista cada segmento com suas coordenadas e raio."""
    print("\n=== Segmentos ===")
    for idx, (a, b) in enumerate(modelo.segmentos):
        pa, pb = modelo.pontos[a], modelo.pontos[b]
        raio = f"{modelo.raios[idx]:.4f}" if modelo.raios is not None else "-"
        print(f"{idx:3d}: {a:3d} -> {b:3d}  "
              f"({pa[0]:+.4f}, {pa[1]:+.4f}) -> ({pb[0]:+.4f}, {pb[1]:+.4f})  raio={raio}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python readVtk.py arquivo.vtk")
        sys.exit(1)

    modelo = ler_vtk(sys.argv[1])
    imprimir_resumo(modelo)
    imprimir_segmentos(modelo)