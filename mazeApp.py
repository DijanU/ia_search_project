import time
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
import random
import streamlit as st
import pandas as pd

# ─── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(page_title="Maze Solver", layout="wide", page_icon="🧩")

st.markdown("""
<style>
    .metric-box { background:#f8f9fa; border-radius:10px; padding:12px 16px; text-align:center; }
    .metric-label { font-size:12px; color:#888; text-transform:uppercase; letter-spacing:1px; }
    .metric-value { font-size:28px; font-weight:600; color:#1a1a2e; }
    .stDataFrame { font-size: 13px; }
    div[data-testid="stMetric"] { background:#f8f9fa; border-radius:10px; padding:10px; }
</style>
""", unsafe_allow_html=True)

# ─── Algoritmos ────────────────────────────────────────────────────────────────

def heuristica(a, b, tipo='manhattan'):
    if tipo == 'manhattan':
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_vecinos(estado, grid):
    rows, cols = len(grid), len(grid[0])
    for dr, dc in [(-1,0),(0,1),(1,0),(0,-1)]:
        nr, nc = estado[0]+dr, estado[1]+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '0':
            yield (nr, nc)

def reconstruir(parent, end):
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]

def bfs(grid, inicio, meta):
    t0 = time.time()
    queue = deque([inicio])
    parent = {inicio: None}
    visited_order = []
    nodos = 0
    while queue:
        cur = queue.popleft()
        visited_order.append(cur)
        nodos += 1
        if cur == meta:
            return reconstruir(parent, cur), nodos, time.time()-t0, visited_order
        for nb in get_vecinos(cur, grid):
            if nb not in parent:
                parent[nb] = cur
                queue.append(nb)
    return None, nodos, time.time()-t0, visited_order

def dfs(grid, inicio, meta):
    t0 = time.time()
    stack = [inicio]
    parent = {inicio: None}
    visited_order = []
    nodos = 0
    while stack:
        cur = stack.pop()
        visited_order.append(cur)
        nodos += 1
        if cur == meta:
            return reconstruir(parent, cur), nodos, time.time()-t0, visited_order
        for nb in reversed(list(get_vecinos(cur, grid))):
            if nb not in parent:
                parent[nb] = cur
                stack.append(nb)
    return None, nodos, time.time()-t0, visited_order

def greedy(grid, inicio, meta, h_tipo='manhattan'):
    t0 = time.time()
    pq = [(heuristica(inicio, meta, h_tipo), inicio)]
    parent = {inicio: None}
    visited_order = []
    nodos = 0
    while pq:
        _, cur = heapq.heappop(pq)
        visited_order.append(cur)
        nodos += 1
        if cur == meta:
            return reconstruir(parent, cur), nodos, time.time()-t0, visited_order
        for nb in get_vecinos(cur, grid):
            if nb not in parent:
                parent[nb] = cur
                heapq.heappush(pq, (heuristica(nb, meta, h_tipo), nb))
    return None, nodos, time.time()-t0, visited_order

def a_star(grid, inicio, meta, h_tipo='manhattan'):
    t0 = time.time()
    pq = [(heuristica(inicio, meta, h_tipo), 0, inicio)]
    parent = {inicio: None}
    g_score = {inicio: 0}
    visited_order = []
    nodos = 0
    while pq:
        f, g, cur = heapq.heappop(pq)
        if g > g_score.get(cur, float('inf')):
            continue
        visited_order.append(cur)
        nodos += 1
        if cur == meta:
            return reconstruir(parent, cur), nodos, time.time()-t0, visited_order
        for nb in get_vecinos(cur, grid):
            ng = g + 1
            if ng < g_score.get(nb, float('inf')):
                g_score[nb] = ng
                parent[nb] = cur
                heapq.heappush(pq, (ng + heuristica(nb, meta, h_tipo), ng, nb))
    return None, nodos, time.time()-t0, visited_order

def calcular_bf(nodos, profundidad):
    if not profundidad or nodos <= 1:
        return 0.0
    return round(nodos ** (1/profundidad), 4)

# ─── Visualización ─────────────────────────────────────────────────────────────

def render_maze(grid, visited=None, path=None, start=None, end=None):
    rows, cols = len(grid), len(grid[0])
    img = np.ones((rows, cols, 3))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '0':
                img[r, c] = [0.1, 0.1, 0.1]       # pared: casi negro
            else:
                img[r, c] = [1.0, 1.0, 1.0]        # pasillo: blanco

    if visited:
        for r, c in visited:
            if grid[r][c] not in ('2', '3'):
                img[r, c] = [0.75, 0.87, 1.0]      # visitado: azul claro

    if path:
        for r, c in path:
            img[r, c] = [0.93, 0.27, 0.27]         # camino: rojo

    if start:
        img[start[0], start[1]] = [0.23, 0.51, 0.96]   # inicio: azul
    if end:
        img[end[0], end[1]] = [0.13, 0.77, 0.37]       # meta: verde

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, interpolation='nearest')
    ax.axis('off')

    patches = [
        mpatches.Patch(color=[0.23,0.51,0.96], label='Inicio'),
        mpatches.Patch(color=[0.13,0.77,0.37], label='Meta'),
        mpatches.Patch(color=[0.75,0.87,1.0],  label='Explorado'),
        mpatches.Patch(color=[0.93,0.27,0.27], label='Camino'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=8,
              framealpha=0.9, ncol=2)
    plt.tight_layout(pad=0)
    return fig

# ─── Cargar laberinto ──────────────────────────────────────────────────────────

def cargar_desde_texto(contenido):
    lines = contenido.strip().split('\n')
    grid = [list(line.strip()) for line in lines if line.strip()]
    return grid

def encontrar_valor(grid, val):
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == val:
                return (r, c)
    return None

# ─── UI Principal ──────────────────────────────────────────────────────────────

st.title("🧩 Maze Solver")
st.caption("Visualizador de algoritmos de búsqueda — BFS · DFS · Greedy · A*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    uploaded = st.file_uploader("Cargar laberinto (.txt)", type="txt")

    st.divider()
    algoritmo = st.selectbox("Algoritmo", [
        "BFS", "DFS",
        "Greedy (Manhattan)", "Greedy (Euclidiana)",
        "A* (Manhattan)", "A* (Euclidiana)",
        "— Todos —"
    ])

    st.divider()
    st.subheader("🎲 Puntos aleatorios")
    num_aleatorios = st.slider("Número de pruebas", 1, 10, 3)
    correr_aleatorios = st.button("Ejecutar simulación aleatoria", use_container_width=True)

    st.divider()
    st.caption("Proyecto #2 · Búsqueda y Heurísticas")

# ─── Estado ───────────────────────────────────────────────────────────────────

if 'grid' not in st.session_state:
    st.session_state.grid = None
    st.session_state.results = []

if uploaded:
    contenido = uploaded.read().decode('utf-8')
    st.session_state.grid = cargar_desde_texto(contenido)
    st.session_state.results = []

grid = st.session_state.grid

if grid is None:
    st.info("👈 Carga un archivo .txt desde el panel izquierdo para comenzar.")
    st.stop()

inicio = encontrar_valor(grid, '2')
meta   = encontrar_valor(grid, '3')

if inicio is None or meta is None:
    st.error("El laberinto debe tener un '2' (inicio) y un '3' (meta).")
    st.stop()

# ─── Mapa de algoritmos ───────────────────────────────────────────────────────

ALGOS = {
    "BFS":               (bfs,    {}),
    "DFS":               (dfs,    {}),
    "Greedy (Manhattan)":(greedy, {'h_tipo':'manhattan'}),
    "Greedy (Euclidiana)":(greedy,{'h_tipo':'euclidiana'}),
    "A* (Manhattan)":    (a_star, {'h_tipo':'manhattan'}),
    "A* (Euclidiana)":   (a_star, {'h_tipo':'euclidiana'}),
}

# ─── Botón principal ──────────────────────────────────────────────────────────

col_btn, _ = st.columns([1, 3])
with col_btn:
    resolver = st.button("▶ Resolver", type="primary", use_container_width=True)

# ─── Ejecutar caso base ───────────────────────────────────────────────────────

if resolver:
    to_run = list(ALGOS.items()) if algoritmo == "— Todos —" else [(algoritmo, ALGOS[algoritmo])]
    st.session_state.results = []

    progress = st.progress(0, text="Ejecutando algoritmos...")
    for i, (nombre, (fn, kwargs)) in enumerate(to_run):
        path, nodos, t, visited = fn(grid, inicio, meta, **kwargs)
        st.session_state.results.append({
            'nombre': nombre,
            'path': path,
            'nodos': nodos,
            'tiempo': t,
            'visited': visited,
        })
        progress.progress((i+1)/len(to_run), text=f"Ejecutando {nombre}...")
    progress.empty()

# ─── Mostrar resultados ───────────────────────────────────────────────────────

if st.session_state.results:
    results = st.session_state.results

    st.subheader("📊 Resultados")

    # Tabla resumen
    import pandas as pd
    rows_data = []
    for r in results:
        length = len(r['path']) if r['path'] else 0
        bf = calcular_bf(r['nodos'], length)
        rows_data.append({
            'Algoritmo': r['nombre'],
            'Longitud': length if length else '—',
            'Nodos explorados': r['nodos'],
            'Tiempo (s)': round(r['tiempo'], 6),
            'Branching Factor': bf,
            'Solución': '✅' if r['path'] else '❌',
        })
    df = pd.DataFrame(rows_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # Visualización por algoritmo
    if len(results) == 1:
        r = results[0]
        length = len(r['path']) if r['path'] else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Longitud", length if length else "—")
        c2.metric("Nodos explorados", f"{r['nodos']:,}")
        c3.metric("Tiempo (s)", f"{r['tiempo']:.6f}")
        c4.metric("Branching Factor", calcular_bf(r['nodos'], length))

        fig = render_maze(grid, r['visited'], r['path'], inicio, meta)
        st.pyplot(fig, use_container_width=True)

    else:
        # Grid de visualizaciones
        st.subheader("🗺️ Visualización por algoritmo")
        for i in range(0, len(results), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i+j >= len(results):
                    break
                r = results[i+j]
                length = len(r['path']) if r['path'] else 0
                with col:
                    st.markdown(f"**{r['nombre']}** — {length} pasos · {r['nodos']:,} nodos · {r['tiempo']:.4f}s")
                    fig = render_maze(grid, r['visited'], r['path'], inicio, meta)
                    st.pyplot(fig, use_container_width=True)

# ─── Simulación aleatoria ─────────────────────────────────────────────────────

if correr_aleatorios and grid is not None:
    st.divider()
    st.subheader(f"🎲 Simulación con {num_aleatorios} puntos de partida aleatorios")

    celdas_libres = [(r, c) for r in range(len(grid))
                     for c in range(len(grid[0]))
                     if grid[r][c] == '0']

    if len(celdas_libres) < num_aleatorios:
        st.warning("No hay suficientes celdas libres.")
    else:
        inicios_rand = random.sample(celdas_libres, num_aleatorios)
        # Ordenar por distancia manhattan a la meta
        inicios_rand.sort(key=lambda p: heuristica(p, meta, 'manhattan'))

        rand_rows = []
        for i, ini in enumerate(inicios_rand):
            dist = heuristica(ini, meta, 'manhattan')
            for nombre, (fn, kwargs) in ALGOS.items():
                path, nodos, t, _ = fn(grid, ini, meta, **kwargs)
                length = len(path) if path else 0
                rand_rows.append({
                    'Prueba': i+1,
                    'Inicio': str(ini),
                    'Dist. Manhattan': int(dist),
                    'Algoritmo': nombre,
                    'Longitud': length if length else '—',
                    'Nodos': nodos,
                    'Tiempo (s)': round(t, 6),
                    'B-Factor': calcular_bf(nodos, length),
                    'Solución': '✅' if path else '❌',
                })

        df_rand = pd.DataFrame(rand_rows)
        st.dataframe(df_rand, use_container_width=True, hide_index=True)

        # Mini mapa del último inicio aleatorio con A*
        st.markdown("**Vista del último punto aleatorio (A* Manhattan):**")
        ultimo_ini = inicios_rand[-1]
        path, nodos, t, visited = a_star(grid, ultimo_ini, meta, h_tipo='manhattan')
        fig = render_maze(grid, visited, path, ultimo_ini, meta)
        st.pyplot(fig, use_container_width=True)