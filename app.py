import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Machine Anomaly Detector", layout="wide", page_icon="💻")

@st.cache_data
def get_data():
    data = pd.read_csv("data_results.csv")
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

df = get_data()

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'run_sim' not in st.session_state:
    st.session_state.run_sim = False

# SIDEBAR
st.sidebar.title("Configuration")
window_size = 500
sim_speed = st.sidebar.select_slider("Vitesse du flux", options=[0.5, 0.1, 0.01], value=0.1)

col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
if col_btn1.button("▶️"):
    st.session_state.run_sim = True
if col_btn2.button("⏸️"):
    st.session_state.run_sim = False
if col_btn3.button("⏹️"):
    st.session_state.current_index = 0
    st.session_state.run_sim = False
    st.rerun()

# HEADER & METRICS
st.title("Monitoring Multi-Modèles")
m1, m2, m3 = st.columns(3)
met1 = m1.empty()
met2 = m2.empty()
met3 = m3.empty()
chart_placeholder = st.empty()

# LOGIQUE DE SIMULATION
while st.session_state.run_sim and st.session_state.current_index < len(df):
    st.session_state.current_index += 5 
    idx = st.session_state.current_index
    
    current_view = df.iloc[max(0, idx - window_size) : idx]
    last_point = current_view.iloc[-1]
    
    # Métriques
    met1.metric("Température", f"{last_point['value']:.1f} °F")
    status = "🚨 ALERTE" if last_point['alert_level'] == 2 else "⚠️ ATTENTION" if last_point['alert_level'] == 1 else "✅ NORMAL"
    met2.metric("Statut Système", status)
    
    total_alerts = int(df.iloc[:idx]['is_anomaly'].fillna(0).sum())
    met3.metric("Alertes", total_alerts)

    fig = go.Figure()
    
    # Couleurs harmonisées pour les pointillés
    models = {
        'i':  {'name': 'IQR', 'color': 'rgba(0, 255, 127, 0.4)'},
        'if': {'name': 'IsoForest', 'color': 'rgba(0, 191, 255, 0.4)'},
        'ae': {'name': 'LSTM-Enc', 'color': 'rgba(255, 165, 0, 0.4)'},
        'p':  {'name': 'Prophet', 'color': 'rgba(255, 0, 255, 0.4)'},
        'l':  {'name': 'LOF', 'color': 'rgba(200, 200, 200, 0.4)'}
    }

    for mod, info in models.items():
        if f'up_{mod}' in current_view.columns:
            # Seuil Haut
            fig.add_trace(go.Scatter(
                x=current_view['timestamp'], y=current_view[f'up_{mod}'], 
                line=dict(color=info['color'], width=1, dash='dot'), 
                name=f"Seuils {info['name']}",
                legendgroup=info['name'],
                showlegend=True
            ))
            # Seuil Bas
            fig.add_trace(go.Scatter(
                x=current_view['timestamp'], y=current_view[f'lo_{mod}'], 
                line=dict(color=info['color'], width=1, dash='dot'), 
                legendgroup=info['name'],
                showlegend=False
            ))

    # Affichage de la Ground Truth (NAB)
    truth_view = current_view[current_view['actual_anomaly'] == 1]

    if not truth_view.empty:
        start_ts = truth_view['timestamp'].min()
        end_ts = truth_view['timestamp'].max()
        
        fig.add_vrect(
            x0=start_ts, x1=end_ts,
            fillcolor="red", 
            opacity=0.2,
            layer="below",
            line_width=0,
            name="Ground Truth (NAB)"
        )

    # Valeur réelle
    fig.add_trace(go.Scatter(x=current_view['timestamp'], y=current_view['value'], 
                             line=dict(color='white', width=2.5), name="Température"))

    # Alerte si consensus (alert_level 2 = 3+ modèles)
    consensus = current_view[current_view['alert_level'] == 2]
    if not consensus.empty:
        fig.add_trace(go.Scatter(x=consensus['timestamp'], y=consensus['value'], 
                                 mode='markers', marker=dict(color='red', size=12, symbol='x'), 
                                 name="ALERTE 🚨"))

    
    fig.update_layout(
        template="plotly_dark", 
        height=650, 
        hovermode="x unified",
        
        xaxis=dict(
            range=[current_view['timestamp'].min(), current_view['timestamp'].max()]
        ),
        
        yaxis=dict(
            range=[df['value'].min() - 5, df['value'].max() + 5]
        ),
        
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)
    time.sleep(sim_speed)

# MODE ANALYSE (PAUSE)
if not st.session_state.run_sim and st.session_state.current_index > 0:
    idx = st.session_state.current_index
    analysis_view = df.iloc[:idx]
    fig_analysis = go.Figure()
    
    fig_analysis.add_trace(go.Scatter(x=analysis_view['timestamp'], y=analysis_view['value'], 
                                     name="Historique", line=dict(color='#00d1b2', width=2)))

    anoms_past = analysis_view[analysis_view['alert_level'] == 2]
    if not anoms_past.empty:
        fig_analysis.add_trace(go.Scatter(x=anoms_past['timestamp'], y=anoms_past['value'], 
                                         mode='markers', marker=dict(color='red', size=8), name="Alertes"))

    fig_analysis.update_layout(
        template="plotly_dark", height=600, 
        dragmode="pan", hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    chart_placeholder.plotly_chart(fig_analysis, use_container_width=True)
    st.info("Mode Analyse")