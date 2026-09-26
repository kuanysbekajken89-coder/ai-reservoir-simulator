import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Smart Reservoir & Economics Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

st.title("🛢️ AI-Powered Digital Oil Field & Economic Simulator")
st.caption("Smart Reservoir Simulation, Infill Drilling Optimization & Economic Viability (West Kazakhstan Pilot Case)")

# Sidebar Controls
st.sidebar.header("⚙️ 1. Reservoir & Fluid Parameters")
grid_size = st.sidebar.slider("Grid Resolution (N x N)", 15, 35, 25, 5)
porosity = st.sidebar.slider("Porosity (ϕ)", 0.10, 0.35, 0.22, 0.01)
viscosity = st.sidebar.slider("Oil Viscosity (cP)", 1.0, 20.0, 8.5, 0.5)

st.sidebar.header("💧 2. EOR & Injection Controls")
injection_rate = st.sidebar.slider("Water Injection Rate (m³/day)", 100, 1000, 600, 50)
sim_steps = st.sidebar.slider("Simulation Time (Months)", 6, 60, 24, 6)

st.sidebar.header("💰 3. Economics & CAPEX/OPEX")
oil_price = st.sidebar.slider("Oil Price ($/barrel)", 40, 120, 75, 5)
capex_per_well = st.sidebar.slider("New Well CAPEX ($ Million)", 1.0, 10.0, 3.5, 0.5)
opex_per_bbl = st.sidebar.slider("OPEX ($/barrel)", 5, 30, 15, 1)

# Simulation Core Function
@st.cache_data
def run_simulation(N, phi, mu, inj_rate, steps):
    np.random.seed(42)
    
    # Heterogeneous Permeability Grid (mD)
    perm = np.random.lognormal(mean=5.0, sigma=0.6, size=(N, N))
    perm = np.clip(perm, 10, 1500)
    
    # Initial Oil Saturation Grid
    So = np.ones((N, N)) * 0.85
    
    # Well Positions
    producers = [(2, 2), (2, N-3), (N-3, 2)]
    injectors = [(N//2, N//2)]
    
    history_So = [So.copy()]
    oil_rates = []
    water_cuts = []
    
    current_So = So.copy()
    
    for step in range(steps):
        mobility = (perm / mu) * (current_So ** 2)
        
        for ix, iy in injectors:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < N and 0 <= ny < N:
                        dist = np.sqrt(dx**2 + dy**2) + 0.1
                        displacement = (inj_rate / 1000) * (mobility[nx, ny] / 50) / dist
                        current_So[nx, ny] = max(0.20, current_So[nx, ny] - displacement * 0.15)
        
        avg_prod_So = np.mean([current_So[px, py] for px, py in producers])
        total_oil_rate = len(producers) * avg_prod_So * (phi / 0.22) * 180 # m3/day
        water_cut = max(0.0, min(95.0, (1 - avg_prod_So / 0.85) * 100 * 1.2))
        
        history_So.append(current_So.copy())
        oil_rates.append(total_oil_rate)
        water_cuts.append(water_cut)
        
    score_grid = current_So * np.log10(perm)
    for px, py in producers + injectors:
        score_grid[max(0, px-1):min(N, px+2), max(0, py-1):min(N, py+2)] = 0
        
    best_idx = np.unravel_index(np.argmax(score_grid), score_grid.shape)
    
    return history_So, oil_rates, water_cuts, best_idx, perm

# Run Sim
history_So, oil_rates, water_cuts, best_well, perm_grid = run_simulation(
    grid_size, porosity, viscosity, injection_rate, sim_steps
)

# Key Performance Indicators (KPIs) & Economics
m3_to_bbl = 6.28981
total_oil_m3 = sum(oil_rates) * 30
total_oil_bbl = total_oil_m3 * m3_to_bbl

revenue = total_oil_bbl * oil_price
opex_total = total_oil_bbl * opex_per_bbl
net_cash_flow = revenue - opex_total - (capex_per_well * 1e6)
roi = (net_cash_flow / (capex_per_well * 1e6)) * 100

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Cumulative Oil Production", f"{total_oil_bbl/1e6:.2f} M bbl")
kpi2.metric("Project Revenue", f"${revenue/1e6:.2f} M")
kpi3.metric("Net Cash Flow (NPV approx)", f"${net_cash_flow/1e6:.2f} M", delta=f"{roi:.1f}% ROI")
kpi4.metric("Avg Final Water Cut", f"{water_cuts[-1]:.1f} %")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🗺️ Reservoir Oil Saturation Map")
    latest_So = history_So[-1]
    
    fig_map = px.imshow(
        latest_So,
        color_continuous_scale="Jet",
        origin="upper",
        labels=dict(color="Oil Saturation (So)"),
        range_color=[0.2, 0.85]
    )
    
    producers = [(2, 2), (2, grid_size-3), (grid_size-3, 2)]
    for px_c, py_c in producers:
        fig_map.add_trace(go.Scatter(x=[py_c], y=[px_c], mode="markers+text", marker=dict(color="black", size=12, symbol="triangle-up"), name="Producer", text=["PROD"], textposition="top center"))
        
    fig_map.add_trace(go.Scatter(x=[grid_size//2], y=[grid_size//2], mode="markers+text", marker=dict(color="cyan", size=14, symbol="square"), name="Injector", text=["INJ"], textposition="top center"))
    fig_map.add_trace(go.Scatter(x=[best_well[1]], y=[best_well[0]], mode="markers+text", marker=dict(color="gold", size=18, symbol="star"), name="AI Rec Well", text=["AI RECOM"], textposition="top center"))
    
    fig_map.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.subheader("📈 Production Dynamics & Economics Flow")
    months = np.arange(1, sim_steps + 1)
    
    fig_dyn = go.Figure()
    fig_dyn.add_trace(go.Scatter(x=months, y=oil_rates, name="Oil Rate (m³/day)", line=dict(color="green", width=3)))
    fig_dyn.add_trace(go.Scatter(x=months, y=water_cuts, name="Water Cut (%)", line=dict(color="blue", width=2, dash="dash"), yaxis="y2"))
    
    fig_dyn.update_layout(
        xaxis=dict(title=dict(text="Time (Months)")),
        yaxis=dict(title=dict(text="Oil Production Rate (m³/day)", font=dict(color="green"))),
        yaxis2=dict(title=dict(text="Water Cut (%)", font=dict(color="blue")), overlaying="y", side="right"),
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_dyn, use_container_width=True)

st.markdown("---")

st.info(f"""
💡 **AI Field Recommendation & Economic Viability Assessment**:
* **Recommended Infill Well Location**: Grid Cell `(X={best_well[1]}, Y={best_well[0]})`
* **Target Zone Oil Saturation**: `{history_So[-1][best_well]*100:.1f}%`
* **Economic Forecast**: At **${oil_price}/bbl**, adding an infill producer at this target yields an estimated **Net Cash Flow of ${net_cash_flow/1e6:.2f} Million** with a return on investment (ROI) of **{roi:.1f}%**.
""")
