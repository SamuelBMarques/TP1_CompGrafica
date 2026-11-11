# TP1 – Visualização Geométrica, Rasterização e Recorte de Árvores Arteriais CCO

Disciplina BCC327 – Computação Gráfica (UFOP, 2026.2) · Prof. Dr. Rafael Alves Bonfim de Queiroz

## 1. Integrantes do grupo

**Grupo XX**

- Samuel Braga Marques – 24.1.4031
- Thiago Martins Zanete – 24.1.4005
- Gabriel Barony Menezes – 24.1.4004

## 2. Tecnologia gráfica utilizada

- **Python 3** com **PyVista (VTK)** para a visualização interativa (Módulos 1 e 2 e a demonstração do recorte na janela 3D).
- **Matplotlib** apenas para *mostrar* as grades de pixels e gerar as figuras dos módulos didáticos (Módulos 3 a 7). Nenhuma função de desenho de linha, preenchimento ou recorte do Matplotlib é usada nos algoritmos.
- **NumPy** para vetores e matrizes.

## 3. Dependências e versões

| Biblioteca | Versão testada | Uso |
|---|---|---|
| Python | 3.12.3 (mínimo 3.10) | linguagem  |
| NumPy | 2.5.4 | cálculo numérico |
| Matplotlib | 3.11.2 | figuras dos módulos didáticos |
| PyVista | 0.49.2 | leitura de VTK, tubos, câmera e janela 3D |


## 4. Instruções de instalação

Não há compilação: é Python puro.

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install numpy matplotlib pyvista
```

## 5. Localização esperada dos arquivos VTK

Os arquivos `.vtk` ficam em `data/` (ou, se não forem incluídos no `.zip`, devem ser obtidos dos pacotes `dados_modelos2D.zip` e colocados nessa pasta).

```
Grupo_XX/
├── src/            código-fonte (os comandos abaixo são executados aqui)
├── data/           arquivos .vtk (ex.: tree2D_Nterm008.vtk)
├── report/         relatorio.tex, relatorio.pdf, figures/
├── presentation/   apresentacao.tex, apresentacao.pdf, figures/
└── README.md
```

Todos os comandos devem ser executados **de dentro de `src/`**. As figuras dos módulos didáticos são salvas automaticamente em `../report/figures/` (a pasta é criada se não existir).

## 6. Como executar

Substitua `../data/arvore.vtk` pelo arquivo escolhido (por exemplo, `../data/tree2D_Nterm008.vtk`).

```bash
cd src

# Aplicação principal (janela interativa: Módulos 1, 2 e demonstração do recorte)
python app.py ../data/arvore.vtk

python visualizador.py ../data/arvore.vtk

# Módulos didáticos (independentes; geram figuras em ../report/figures)
python bresenham.py        ../data/arvore.vtk      # Módulo 3
python scanlineFill.py     ../data/arvore.vtk      # Módulo 4
python antiAliasing.py     ../data/arvore.vtk      # Módulo 5
python recorteSegmentos.py ../data/arvore.vtk      # Módulo 6 (algoritmos + testes + exemplo)
python recorteArvore.py    ../data/arvore.vtk      # Módulo 6 (recorte da árvore, saída em pixels)
python recortePoligonos.py ../data/arvore.vtk      # Módulo 7
```

Opções úteis:

```bash
python app.py ../data/arvore.vtk --projecao ortografica          # projeção inicial: perspectiva (padrão) ou ortografica
python app.py ../data/arvore.vtk --recorte --algoritmo 1          # inicia com o recorte (0 = Cohen-Sutherland, 1 = Liang-Barsky)
python app.py ../data/arvore.vtk --janela XMIN XMAX YMIN YMAX     # janela de recorte em coordenadas do mundo
python app.py ../data/arvore.vtk --modo 1 --vista isometrica --rotx 10 --roty 10 --projecao ortografica --captura fig.png
                                                                  # salva imagem sem abrir janela (vista: xy, xz, yz, isometrica ou 0-3;
                                                                  # --rotx/--roty/--rotz em graus; --projecao perspectiva|ortografica)

python recorteArvore.py    ../data/arvore.vtk --janela XMIN XMAX YMIN YMAX
python recorteArvore.py    ../data/arvore.vtk --tamanho 0.4 --n 128 --vetorial
python recortePoligonos.py                                        # sem arquivo: só as demonstrações didáticas
python visualizador.py     ../data/arvore.vtk                     # visualizador básico (Módulos 1 e 2)
python readVtk.py          ../data/arvore.vtk                     # leitor próprio: imprime resumo e segmentos
```

Se a janela não for informada, `recorteArvore.py`, `recortePoligonos.py` e `app.py` escolhem automaticamente uma janela que corta vários segmentos da árvore lida.

## 7. Comandos de teclado e mouse (`app.py`)

**Mouse** (padrão do PyVista): botão esquerdo gira a câmera, roda faz zoom, botão do meio (ou Shift + esquerdo) move a câmera.

| Tecla | Ação |
|---|---|
| `m` | alterna o modo: linhas coloridas pelo raio / tubos (espessura + cor) / tubos (só espessura) |
| `v` | próxima vista do observador: xy, xz, yz, isométrica |
| `o` | alterna projeção ortográfica / perspectiva |
| setas | translação do modelo |
| `+` / `-` | escala do modelo |
| `z` `x` | rotação em Z (+10° / −10°) |
| `a` `d` | rotação em Y (+10° / −10°) |
| `u` `i` | rotação em X (+10° / −10°) |
| `0` | restaura as transformações |
| `1` | liga/desliga o recorte (vale em linhas e em tubos) |
| `2` | alterna Cohen–Sutherland / Liang–Barsky |
| `h` `j` `k` `l` | move a janela de recorte (esquerda / baixo / cima / direita) |
| `,` `.` | reduz / amplia a janela de recorte |

Rotação e escala são feitas em torno do centro do modelo. O recorte acompanha as transformações aplicadas ao modelo.

## 8. Roteiro mínimo de teste

1. **Módulo 1 – leitura e atributos.** `python readVtk.py ../data/arvore.vtk` imprime número de pontos e segmentos, raio mínimo e máximo, limites e quantidade de folhas.
2. **Módulos 1 e 2 – visualizador.** `python app.py ../data/arvore.vtk`. Teste `m` (os três modos, com barra de cores de raio no modo 1), `v` (4 vistas), `o` (projeção), setas, `+`/`-`, `z`/`x`/`a`/`d`/`u`/`i` e `0`.
3. **Módulo 3 – Bresenham.** `python bresenham.py ../data/arvore.vtk`. Deve imprimir `Todos os testes de Bresenham passaram.`, a tabela do exemplo (0,0)→(8,3) e gerar `bresenham_exemplo.png` e `bresenham_arvore.png`.
4. **Módulo 4 – Scanline Fill.** `python scanlineFill.py ../data/arvore.vtk`. Deve imprimir `Scanline confere com o teste par-impar em 202 poligonos.` e `Pixels do U: 222 (area esperada 222)`, e gerar `scanline_fill.png` e `scanline_extremos.png`.
5. **Módulo 5 – aliasing × antialiasing.** `python antiAliasing.py ../data/arvore.vtk`. Imprime o erro quadrático médio para k = 1, 2, 4, 8 e gera `antialiasing.png` (com e sem supersampling, com detalhe ampliado).
6. **Módulo 6 – recorte de segmentos.** `python recorteSegmentos.py ../data/arvore.vtk`. Deve imprimir `Recorte: 11 casos fixos e 40000 segmentos aleatorios conferidos`, o exemplo passo a passo dos dois algoritmos e gerar `recorte_exemplo.png` e `recorte_arvore.png`. Depois rode `recorteArvore.py` para a saída em grade de pixels.
7. **Recorte na janela 3D.** No `app.py`, tecla `1` para ligar o recorte, `2` para trocar o algoritmo, `h j k l` e `,` `.` para mover/redimensionar a janela, `m` para ver o recorte também nos tubos.
8. **Módulo 7 – recorte de polígonos.** `python recortePoligonos.py ../data/arvore.vtk`. Deve imprimir `Sutherland-Hodgman: 4 casos fixos e 300 poligonos aleatorios conferidos pixel a pixel` e gerar `recorte_poligono_exemplo.png`, `recorte_poligono_concavo.png` e `recorte_poligono_arvore.png`.

## 9. Correspondência com o enunciado

| Módulo do enunciado | Arquivos | Figuras geradas |
|---|---|---|
| 1 – Leitura, primitivas e atributos | `readVtk.py`, `readVtk_PyVista.py`, `visualizador.py`, `app.py` | capturas com `--captura` |
| 2 – Transformações e observador | `visualizador.py`, `app.py` | capturas com `--captura` |
| 3 – Rasterização de linhas (Bresenham) | `bresenham.py` | `bresenham_*.png` |
| 4 – Rasterização de polígonos (Scanline) | `scanlineFill.py` | `scanline_*.png` |
| 5 – Aliasing e antialiasing (SSAA) | `antiAliasing.py` | `antialiasing.png` |
| 6 – Recorte de segmentos | `recorteSegmentos.py`, `recorteArvore.py`, `app.py` | `recorte_exemplo.png`, `recorte_arvore.png` |
| 7 – Recorte de polígonos | `recortePoligonos.py` | `recorte_poligono_*.png` |
| Apoio | `util.py` (leitura, mapeamento mundo→pixel, folhas e raiz) | – |

## 10. O que foi implementado pelo grupo e o que usa biblioteca

### Implementado pelo grupo

- **Leitor de VTK legado ASCII** (`readVtk.py`), sem depender de VTK/PyVista.
- **Matrizes de transformação 4×4** em coordenadas homogêneas (translação, rotação em X, Y e Z, escala uniforme) e sua composição em torno do centro do modelo, `T(c)·R·T(−c)`.
- **Mapeamento mundo → pixel** (`criar_mapeamento`: escala uniforme e translação).
- **Bresenham** (Módulo 3), com verificação automática das propriedades do algoritmo.
- **Scanline Fill** (Módulo 4), com regra par-ímpar e validação contra um teste independente (lançamento de raio) em 202 polígonos; polígono formado por raiz e folhas da árvore.
- **Supersampling (SSAA)** (Módulo 5), usando o Bresenham e o Scanline do grupo, com filtro de caixa e cálculo do erro quadrático médio contra uma referência 16×16.
- **Cohen–Sutherland** e **Liang–Barsky** (Módulo 6), com testes automáticos (os dois algoritmos concordam, extremos dentro da janela, trecho maximal) e comparação de custo.
- **Sutherland–Hodgman** (Módulo 7), validado pixel a pixel contra o teste par-ímpar, com demonstração em polígono côncavo.
- **Escolha automática da janela de recorte** e integração do recorte ao visualizador.

### Realizado com auxílio de bibliotecas

- **PyVista/VTK:** leitura do arquivo (`pv.read`), geração dos tubos (`tube`), mapa de cores *viridis* e barra de cores, janela, câmera, vistas (`view_xy`, `view_xz`, `view_yz`, `view_isometric`), projeção ortográfica/perspectiva.
- **Aplicação das matrizes ao desenho:** as matrizes são escritas pelo grupo, mas quem as multiplica pelos vértices na hora de renderizar é o VTK (via `user_matrix` do ator). A geometria original não é alterada.
- **Matplotlib:** exibição das grades de pixels, janelas de recorte e figuras (`imshow`, linhas e retângulos para exibição).
- **NumPy:** operações vetoriais e matriciais.

### Apenas discutidos (não implementados)

- **Cyrus–Beck** (Módulo 6) e **Weiler–Atherton** (Módulo 7): a diferença em relação aos algoritmos implementados e as situações de uso estão discutidas no relatório e nos comentários finais de `recorteSegmentos.py` e `recortePoligonos.py`.

## 11. Observações e limitações

- Os Módulos 3 a 7 trabalham sobre a **projeção 2D (plano xy)** da árvore, como permitido pelo enunciado. Em modelos 3D, o `z` é descartado nesses módulos. No `app.py`, o recorte também é feito na projeção xy, e o `z` de cada extremo recortado é interpolado ao longo do segmento.
- A comparação entre projeção ortográfica e perspectiva é feita no visualizador principal.
- O supersampling é demonstrado no módulo didático (`antiAliasing.py`), não no visualizador interativo.
- O visualizador aceita apenas células de 2 pontos (segmentos), como nos modelos fornecidos, e a escala é uniforme.
- Os arquivos VTK em formato binário não são lidos pelo leitor próprio `readVtk.py`.