
# streamlit_app_backward_simple.py  (v3 – renamed UI text)
# Backward de‑angulation visualizer (simple UI, Plotly animation)
# Inputs are *projected* angulations from AP/Lateral X‑rays and apparent torsion from axial CT.
# The app solves for the true axial twist that reproduces the measured apparent torsion,
# then de‑angulates to reveal the residual (true) torsion.
#
# Run:
#   pip install streamlit numpy plotly
#   streamlit run streamlit_app_backward_simple.py

import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ------------------ Math helpers ------------------
def rot_x(a):
    t=np.deg2rad(a); c,s=np.cos(t),np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(b):
    t=np.deg2rad(b); c,s=np.cos(t),np.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(g):
    t=np.deg2rad(g); c,s=np.cos(t),np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def normalize(v):
    n=np.linalg.norm(v); 
    return v if n == 0 else v/n

def angle_xy(u, v):
    u2 = np.array([u[0], u[1]]); v2 = np.array([v[0], v[1]])
    if np.linalg.norm(u2) < 1e-9 or np.linalg.norm(v2) < 1e-9: return float('nan')
    u2 = u2/np.linalg.norm(u2); v2 = v2/np.linalg.norm(v2)
    dot = float(np.clip(u2 @ v2, -1.0, 1.0))
    det = float(u2[0]*v2[1] - u2[1]*v2[0])
    return np.rad2deg(np.arctan2(det, dot))

# ------------------ Geometry helpers ------------------
def cylinder_between(z0, z1, radius=0.3, n_theta=40, n_z=22):
    thetas = np.linspace(0, 2*np.pi, n_theta)
    zs = np.linspace(z0, z1, n_z)
    T, Z = np.meshgrid(thetas, zs)
    x = radius * np.cos(T); y = radius * np.sin(T); z = Z
    return x, y, z

def transform_points(R, pts):
    return (R @ pts.T).T

def anterior_stripe_lines(z0, z1, radius=0.3, theta=np.pi/2, n=60):
    zs = np.linspace(z0, z1, n)
    xs = radius*np.cos(theta)*np.ones_like(zs)
    ys = radius*np.sin(theta)*np.ones_like(zs)
    return np.stack([xs, ys, zs], axis=1)

def bone_traces_arrays(R_dist, limit=3.0):
    # Prox cylinder/stripe (fixed)
    xp, yp, zp = cylinder_between(0.0, 1.5, 0.3)
    sp = anterior_stripe_lines(0.0, 1.5, 0.3)
    # Dist cylinder/stripe (rotated)
    xd, yd, zd = cylinder_between(0.0, -1.5, 0.3)
    pts_d = np.stack([xd.flatten(), yd.flatten(), zd.flatten()], axis=1)
    td = transform_points(R_dist, pts_d)
    xd2 = td[:,0].reshape(xd.shape); yd2 = td[:,1].reshape(yd.shape); zd2 = td[:,2].reshape(zd.shape)
    sd = anterior_stripe_lines(0.0, -1.5, 0.3)
    tsd = transform_points(R_dist, sd)
    # Vectors
    origin = np.zeros(3); Zp = np.array([0,0,1]); Ap = np.array([0,1,0])
    Zd = normalize(R_dist @ np.array([0,0,-1])); Ad = normalize(R_dist @ np.array([0,1,0]))
    # XY floor and projections
    rng=np.linspace(-limit, limit, 2); Xf,Yf=np.meshgrid(rng, rng); Zf=np.full_like(Xf,-limit)
    zproj = -limit + 1e-3
    Aprox_xy = np.array([[0,0,zproj],[Ap[0]*1.2, Ap[1]*1.2, zproj]])
    Adist_xy = np.array([[0,0,zproj],[Ad[0]*1.2, Ad[1]*1.2, zproj]])
    return {
        "xp":xp, "yp":yp, "zp":zp, "sp":sp,
        "xd":xd2, "yd":yd2, "zd":zd2, "sd":tsd,
        "Zp":np.stack([origin, Zp], axis=0),
        "Zd":np.stack([origin, Zd], axis=0),
        "Ap":np.stack([origin, Ap*1.2], axis=0),
        "Ad":np.stack([origin, Ad*1.2], axis=0),
        "Xf":Xf, "Yf":Yf, "Zf":Zf, "Aprox_xy":Aprox_xy, "Adist_xy":Adist_xy
    }

# --- Solve for true twist that yields the specified *apparent* torsion ---
def solve_true_twist_from_apparent(alpha, beta, phi_app_deg):
    A = np.array([0,1,0])
    best_gamma = 0.0
    best_err = 1e9
    for g in np.linspace(-180, 180, 1441):  # 0.25° coarse
        R = rot_y(beta) @ rot_x(alpha) @ rot_z(g)
        Ad = normalize(R @ A)
        phi = angle_xy(A, Ad)
        err = abs(((phi - phi_app_deg + 180) % 360) - 180)
        if err < best_err:
            best_err = err; best_gamma = g
    # refine ±1°
    for g in np.linspace(best_gamma-1.0, best_gamma+1.0, 41):
        R = rot_y(beta) @ rot_x(alpha) @ rot_z(g)
        Ad = normalize(R @ A)
        phi = angle_xy(A, Ad)
        err = abs(((phi - phi_app_deg + 180) % 360) - 180)
        if err < best_err:
            best_err = err; best_gamma = g
    return best_gamma

# ------------------ App ------------------
st.set_page_config(page_title="True Torsion Revealer", layout="wide")
left, right = st.columns([1.65, 0.35], gap="large")

with left:
    st.markdown("<h1 style='margin-bottom:0'>True Torsion Revealer – Correcting for Projection Errors</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#555;margin-top:2px;margin-bottom:16px'>"
                "Distinguish between what looks like torsion on X‑rays and CT slices, "
                "and the <b>true residual twist</b> that remains after de‑angulation."
                "</div>", unsafe_allow_html=True)

with right:
    st.markdown("### Inputs")
    st.caption("Enter <b>projected</b> measurements: <br>"
               "• <b>Coronal</b> from <b>AP X‑ray</b> (about Y) <br>"
               "• <b>Sagittal</b> from <b>Lateral X‑ray</b> (about X) <br>"
               "• <b>Apparent torsion</b> from <b>Axial CT</b> (about Z)", unsafe_allow_html=True)
    # order: sagittal, coronal, torsion — but each line explicitly states source/view
    alpha = st.number_input("Sagittal angulation (from Lateral X‑ray, about X) [deg]", -90, 90, 0, step=1, format="%d")
    beta  = st.number_input("Coronal angulation (from AP X‑ray, about Y) [deg]",    -90, 90, 0, step=1, format="%d")
    phi_app = st.number_input("Apparent torsion (from Axial CT, about Z) [deg]",   -180, 180, 0, step=1, format="%d")
    st.markdown("---")
    show_bone  = st.checkbox("Show bone", value=True)
    show_stick = st.checkbox("Show stick", value=True)
    show_xyproj = st.checkbox("Show XY floor", value=True)

# Convert apparent torsion to the true axial twist that produces it
gamma_true = solve_true_twist_from_apparent(alpha, beta, phi_app)

# Build initial posture with this true twist
R_init = rot_y(beta) @ rot_x(alpha) @ rot_z(gamma_true)
limit=3.0

# Residual true torsion after full de‑angulation
R_final = rot_x(-alpha) @ rot_y(-beta) @ R_init
A_prox = np.array([0,1,0]); A_dist_final = normalize(R_final @ A_prox)
true_torsion = angle_xy(A_prox, A_dist_final)

with left:
    st.markdown("<div style='font-size:22px;color:#333;margin-top:4px'>True torsion (residual after de‑angulation)</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:44px;font-weight:700;margin-top:-6px;margin-bottom:8px'>{true_torsion:.1f}°</div>", unsafe_allow_html=True)

# ---- Plotly animation: de‑angulate from R_init to zero tilt ----
def traces_from_arrays(arr, show_bone, show_stick, show_xyproj):
    tr = []
    if show_bone:
        tr.append(go.Surface(x=arr['xp'], y=arr['yp'], z=arr['zp'], showscale=False,
                             colorscale=[[0,'#d9e4ff'],[1,'#d9e4ff']], name='prox'))
        tr.append(go.Scatter3d(x=arr['sp'][:,0], y=arr['sp'][:,1], z=arr['sp'][:,2], mode='lines',
                               line=dict(color='crimson', width=10), name='prox_stripe'))
        tr.append(go.Surface(x=arr['xd'], y=arr['yd'], z=arr['zd'], showscale=False,
                             colorscale=[[0,'#ffe0cc'],[1,'#ffe0cc']], name='dist'))
        tr.append(go.Scatter3d(x=arr['sd'][:,0], y=arr['sd'][:,1], z=arr['sd'][:,2], mode='lines',
                               line=dict(color='crimson', width=10), name='dist_stripe'))
    if show_stick:
        tr.append(go.Scatter3d(x=arr['Zp'][:,0], y=arr['Zp'][:,1], z=arr['Zp'][:,2], mode='lines',
                               line=dict(color='royalblue', width=12), name='Z_prox'))
        tr.append(go.Scatter3d(x=arr['Zd'][:,0], y=arr['Zd'][:,1], z=arr['Zd'][:,2], mode='lines',
                               line=dict(color='darkorange', width=12), name='Z_dist'))
        tr.append(go.Scatter3d(x=arr['Ap'][:,0], y=arr['Ap'][:,1], z=arr['Ap'][:,2], mode='lines',
                               line=dict(color='seagreen', width=12), name='A_prox'))
        tr.append(go.Scatter3d(x=arr['Ad'][:,0], y=arr['Ad'][:,1], z=arr['Ad'][:,2], mode='lines',
                               line=dict(color='crimson', width=12), name='A_dist'))
    if show_xyproj:
        tr.append(go.Surface(x=arr['Xf'], y=arr['Yf'], z=arr['Zf'], showscale=False, opacity=0.10, name='XY floor'))
        tr.append(go.Scatter3d(x=arr['Aprox_xy'][:,0], y=arr['Aprox_xy'][:,1], z=arr['Aprox_xy'][:,2], mode='lines',
                               line=dict(color='rgba(46,139,87,0.65)', width=6), name='Aprox→XY'))
        tr.append(go.Scatter3d(x=arr['Adist_xy'][:,0], y=arr['Adist_xy'][:,1], z=arr['Adist_xy'][:,2], mode='lines',
                               line=dict(color='rgba(220,20,60,0.75)', width=6), name='Adist→XY'))
    return tr

arr0 = bone_traces_arrays(R_init, limit=limit)

frames = []
n_frames = 60
for i in range(n_frames+1):
    t = i / n_frames
    R_t = rot_x(-alpha * t) @ rot_y(-beta * t) @ R_init
    arr_t = bone_traces_arrays(R_t, limit=limit)
    frames.append(go.Frame(data=traces_from_arrays(arr_t, show_bone, show_stick, show_xyproj), name=f"{t:.3f}"))

fig = go.Figure(data=traces_from_arrays(arr0, show_bone, show_stick, show_xyproj))
fig.frames = frames

sliders = [dict(active=0, steps=[dict(method='animate',
                                      args=[[f"{i/n_frames:.3f}"],
                                            {'mode':'immediate','frame':{'duration':0,'redraw':True},'transition':{'duration':0}}],
                                      label=f"{i/n_frames:.2f}") for i in range(n_frames+1)],
                x=0.10, y=0.04, len=0.78,
                currentvalue=dict(prefix='Progress: ', suffix='  (0→1)'))]

updatemenus=[dict(type='buttons', showactive=False, x=0.10, y=0.10,
                  buttons=[dict(label='Play', method='animate',
                                args=[None, {'fromcurrent':True,'frame':{'duration':35,'redraw':True},'transition':{'duration':0}}]),
                           dict(label='Pause', method='animate',
                                args=[[None], {'mode':'immediate'}])])]

fig.update_layout(scene=dict(xaxis=dict(range=[-3,3], title='X'),
                             yaxis=dict(range=[-3,3], title='Y'),
                             zaxis=dict(range=[-3,3], title='Z'),
                             aspectmode='cube',
                             bgcolor='rgb(248,248,248)'),
                  paper_bgcolor='white',
                  margin=dict(l=0,r=0,t=10,b=0),
                  height=780,
                  sliders=sliders, updatemenus=updatemenus)

with left:
    st.plotly_chart(fig, use_container_width=True)
