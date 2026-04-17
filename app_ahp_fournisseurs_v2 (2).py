"""
============================================================
Système de Pilotage de la Performance Fournisseur
Méthode AHP (Saaty, 1980) + K-means (MacQueen, 1967)
PFE 2025-2026
============================================================
RÉVISION v2 :
  - Nombre de critères configurable (3 à 8)
  - K-means fiabilisé : initialisation par percentiles AHP
    (supprime l'instabilité aléatoire de k-means++)
  - Seuils Tier configurables (distribution Pareto maîtrisée)
============================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import io

# ── Configuration page ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AHP – Segmentation Fournisseurs",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-banner {
        background: linear-gradient(135deg, #1b2a4a 0%, #2e5090 100%);
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        border-left: 6px solid #c9a84c;
    }
    .main-banner h1 { color: #ffffff; font-size: 1.6rem; margin: 0 0 6px 0; }
    .main-banner p  { color: #c9a84c; font-size: 0.9rem; margin: 0; }
    .tier-card {
        padding: 20px; border-radius: 10px; text-align: center;
        font-weight: bold; font-size: 1.1rem; margin: 4px;
    }
    .tier1 { background:#d4edda; color:#1e7e34; border: 2px solid #1e7e34; }
    .tier2 { background:#fff3cd; color:#856404; border: 2px solid #856404; }
    .tier3 { background:#f8d7da; color:#721c24; border: 2px solid #721c24; }
    .metric-box {
        background: #f0f4ff; border-radius: 8px;
        padding: 12px 16px; border-left: 4px solid #2e5090; margin: 6px 0;
    }
    .cr-ok  { background:#d4edda; color:#155724; padding:6px 14px;
              border-radius:20px; font-weight:bold; font-size:0.95rem; }
    .cr-ko  { background:#f8d7da; color:#721c24; padding:6px 14px;
              border-radius:20px; font-weight:bold; font-size:0.95rem; }
    .gold-sep { height:3px; background:#c9a84c; border-radius:2px; margin: 18px 0; }
    .info-box {
        background:#e8f4fd; border-left:4px solid #2e5090;
        border-radius:6px; padding:10px 14px; margin:8px 0;
        font-size:0.88rem; color:#1a3a5c;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,
            5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

DEFAULT_NAMES = {
    3: ["CA HT", "Volume", "Criticité"],
    4: ["CA HT", "Volume", "Rentabilité", "Criticité"],
    5: ["CA HT", "Volume", "Rentabilité", "Criticité", "Qualité"],
    6: ["CA HT", "Volume", "Rentabilité", "Criticité", "Qualité", "Délai"],
    7: ["CA HT", "Volume", "Rentabilité", "Criticité", "Qualité", "Délai", "Fiabilité"],
    8: ["CA HT", "Volume", "Rentabilité", "Criticité", "Qualité", "Délai", "Fiabilité", "Innovation"],
}

DEFAULT_AHP_VALS = {(0,1):3.0, (0,2):0.5, (0,3):1.0,
                    (1,2):0.25, (1,3):1/3, (2,3):2.0}

TIER_COLORS = {"Tier 1":"#1e7e34", "Tier 2":"#f0a500", "Tier 3":"#dc3545"}
TIER_BG     = {"Tier 1":"#d4edda", "Tier 2":"#fff3cd", "Tier 3":"#f8d7da"}


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS AHP  (génériques — n quelconque)
# ══════════════════════════════════════════════════════════════════════════════

def get_pairs(n: int) -> list:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def build_matrix(vals: list, n: int) -> np.ndarray:
    A = np.eye(n)
    for idx, (i, j) in enumerate(get_pairs(n)):
        A[i, j] = vals[idx]
        A[j, i] = 1.0 / vals[idx]
    return A


def ahp_weights(A: np.ndarray):
    n       = A.shape[0]
    col_sum = A.sum(axis=0)
    w       = (A / col_sum).mean(axis=1)
    Aw      = A @ w
    lam_max = (Aw / w).mean()
    CI      = (lam_max - n) / (n - 1)
    RI      = RI_TABLE.get(n, 1.49)
    CR      = CI / RI if RI > 0 else 0.0
    return w, lam_max, CI, CR


def interpret_saaty(val: float, ci: str, cj: str) -> str:
    if abs(val - 1) < 1e-9:
        return "Égale importance"
    labels = {9: "Absolue", 7: "Très forte", 5: "Forte", 3: "Modérée"}
    for thr, lbl in labels.items():
        if val >= thr:
            return f"{lbl} faveur → {ci}"
        if val <= 1 / thr:
            return f"{lbl} faveur → {cj}"
    return f"Légère faveur → {'→ '+ci if val > 1 else '→ '+cj}"


# ══════════════════════════════════════════════════════════════════════════════
# MOTEUR DE SEGMENTATION  (deux méthodes fiables sur le score AHP 1D)
# ══════════════════════════════════════════════════════════════════════════════

def segment_direct(scores: np.ndarray, p_tier1: int = 80, p_tier3: int = 45) -> tuple:
    """
    Méthode A — Affectation directe par seuils percentiles (recommandée).

    Principe
    ────────
    AHP agrège déjà n critères en un score 1D parfaitement ordonné.
    Il suffit de découper ce score en trois zones par deux seuils percentiles :

        Score ≥ percentile(p_tier1)  → Tier 1 (Stratégiques)
        Score <  percentile(p_tier3) → Tier 3 (Basiques)
        Sinon                        → Tier 2 (Essentiels)

    Avantages
    ─────────
    • Distribution exactement contrôlée (pas d'aléatoire).
    • Stable : même résultat à chaque exécution.
    • Transparente : chaque seuil est interprétable par le métier.
    • Robuste quelle que soit la distribution des données (Pareto, normale, etc.)

    Paramètres
    ----------
    scores   : score AHP 1D = Z @ w  (N,)
    p_tier1  : percentile → seuil supérieur [défaut 80 = top 20% en Tier 1]
    p_tier3  : percentile → seuil inférieur [défaut 45 = bottom 45% en Tier 3]

    Retourne
    --------
    labels        : 0=Tier1, 1=Tier2, 2=Tier3  (N,)
    thr1, thr3    : seuils de score appliqués
    cluster_means : score moyen [Tier1, Tier2, Tier3]
    """
    thr1 = np.percentile(scores, p_tier1)
    thr3 = np.percentile(scores, p_tier3)

    labels = np.where(scores >= thr1, 0,
             np.where(scores <  thr3, 2, 1))

    cluster_means = np.array([
        scores[labels == k].mean() if (labels == k).any() else 0.0
        for k in range(3)
    ])
    return labels, thr1, thr3, cluster_means


def segment_kmeans1d(scores: np.ndarray, p_tier1: int = 80, p_tier3: int = 45,
                     random_state: int = 42) -> tuple:
    """
    Méthode B — K-means 1D sur le score AHP (natural breaks).

    Principe
    ────────
    K-means est appliqué sur le score AHP 1D (et non dans l'espace n-D).
    C'est la seule utilisation valide de K-means ici car :
    - Le score AHP est déjà la synthèse pondérée de tous les critères.
    - En 1D, K-means trouve des "natural breaks" (similaire à Jenks/Fisher).
    - L'initialisation par percentiles garantit la convergence Pareto.

    Pourquoi PAS en n-D ?
    ─────────────────────
    En espace n-D standardisé, chaque dimension suit N(0,1). La somme de
    n variables N(0,1) → N(0,n) par le TCL. K-means sur cette distribution
    symétrique converge toujours vers 3 clusters quasi-égaux (~33/33/33),
    quelle que soit l'initialisation.

    Paramètres
    ----------
    scores   : score AHP 1D = Z @ w  (N,)
    p_tier1  : percentile → centroïde initial Tier 1
    p_tier3  : percentile → centroïde initial Tier 3
    """
    thr1 = np.percentile(scores, p_tier1)
    thr3 = np.percentile(scores, p_tier3)

    m1 = scores >= thr1
    m3 = scores < thr3
    m2 = ~m1 & ~m3

    # Centroïdes initiaux sur le score 1D
    c1_init = scores[m1].mean() if m1.any() else scores.max()
    c3_init = scores[m3].mean() if m3.any() else scores.min()
    c2_init = scores[m2].mean() if m2.any() else (c1_init + c3_init) / 2
    init    = np.array([[c1_init], [c2_init], [c3_init]])

    km = KMeans(n_clusters=3, init=init, n_init=1,
                max_iter=500, random_state=random_state)
    km.fit(scores.reshape(-1, 1))

    # Tri : cluster 0 = score le plus élevé
    c_means = np.array([scores[km.labels_ == k].mean() for k in range(3)])
    order   = np.argsort(-c_means)
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    labels  = np.array([mapping[c] for c in km.labels_])

    cluster_means = c_means[order]
    breaks = sorted([scores[labels == k].min() for k in [0, 1]])  # seuils naturels
    return labels, breaks[0], breaks[1], cluster_means


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-banner">
  <h1>📦 Système de Pilotage de la Performance Fournisseur</h1>
  <p>Segmentation AHP (Saaty, 1980) + K-means fiabilisé · Critères configurables (3 à 8)</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ Navigation")
    step = st.radio("Étape", [
        "① Définir les critères",
        "② Matrice AHP",
        "③ Importer les données",
        "④ Résultats & Tiers",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Échelle de Saaty**")
    st.dataframe(pd.DataFrame({
        "Valeur": [1, 3, 5, 7, 9, "1/3…1/9"],
        "Signification": ["Égale", "Modérée", "Forte", "Très forte", "Absolue", "Réciproques"]
    }), hide_index=True, use_container_width=True)

    if step == "④ Résultats & Tiers":
        st.markdown("---")
        st.markdown("### ⚙️ Paramètres de segmentation")

        method = st.radio(
            "Méthode de segmentation",
            ["Seuils directs (recommandé)", "K-means 1D (natural breaks)"],
            help=(
                "**Seuils directs** : affectation par percentiles du score AHP. "
                "Distribution exactement contrôlée, résultat garanti.\n\n"
                "**K-means 1D** : trouve les ruptures naturelles dans la distribution "
                "du score AHP. Utile si les fournisseurs forment des groupes distincts."
            )
        )
        st.session_state["seg_method"] = method

        st.markdown("""
        <div class="info-box">
        Les seuils s'appliquent au <b>score AHP 1D</b> (= Z·w),
        seule dimension sur laquelle la segmentation est mathématiquement fiable.
        </div>
        """, unsafe_allow_html=True)

        p_t1 = st.slider("Seuil Tier 1 (percentile AHP)", 60, 90, 80, 5,
            help="Score ≥ ce percentile → Tier 1. Ex : 80 = top 20% en Tier 1.")
        p_t3 = st.slider("Seuil Tier 3 (percentile AHP)", 20, 60, 45, 5,
            help="Score < ce percentile → Tier 3. Ex : 45 = bottom 45% en Tier 3.")

        if p_t3 >= p_t1:
            st.error("⚠️ Seuil Tier 3 doit être < Seuil Tier 1.")
        else:
            pct1 = 100 - p_t1
            pct3 = p_t3
            pct2 = 100 - pct1 - pct3
            st.markdown("**Distribution Pareto cible :**")
            st.markdown(f"🟢 Tier 1 ≈ **{pct1}%** · 🟡 Tier 2 ≈ **{pct2}%** · 🔴 Tier 3 ≈ **{pct3}%**")
            st.caption("Note : K-means 1D peut légèrement dévier selon les ruptures naturelles.")

        st.session_state["km_p_tier1"] = p_t1
        st.session_state["km_p_tier3"] = p_t3

    st.markdown("---")
    st.caption("PFE 2025-2026 · AHP + K-means fiabilisé")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Définition des critères
# ══════════════════════════════════════════════════════════════════════════════
if step == "① Définir les critères":
    st.subheader("ÉTAPE 1 — Définir les critères d'évaluation")

    st.markdown("#### Nombre de critères")
    st.markdown("""
    <div class="info-box">
    Le nombre de critères détermine la taille de la matrice AHP (n×n) et le nombre
    de comparaisons par paires : <b>n·(n-1)/2</b>.<br>
    ▸ 3 → 3 cmp &nbsp;|&nbsp; 4 → 6 &nbsp;|&nbsp; 5 → 10 &nbsp;|&nbsp;
      6 → 15 &nbsp;|&nbsp; 7 → 21 &nbsp;|&nbsp; 8 → 28<br>
    Au-delà de 6 critères, maintenir CR &lt; 10% devient difficile.
    </div>
    """, unsafe_allow_html=True)

    n_prev  = st.session_state.get("n_crit", 4)
    n_crit  = st.selectbox(
        "Nombre de critères",
        options=[3, 4, 5, 6, 7, 8],
        index=[3, 4, 5, 6, 7, 8].index(n_prev),
        format_func=lambda n: f"{n} critères — {n*(n-1)//2} comparaisons",
        key="n_crit_select",
    )

    if n_crit != n_prev:
        for k in list(st.session_state.keys()):
            if k.startswith("aij_") or k in [
                "ahp_weights","ahp_lam","ahp_CI","ahp_CR","ahp_names",
                "ahp_matrix","df_fournisseurs","df_results"
            ]:
                del st.session_state[k]
        st.info(f"Critères modifiés → {n_crit}. Les étapes suivantes ont été réinitialisées.")
    st.session_state["n_crit"] = n_crit

    st.markdown("---")
    st.markdown(f"#### Nommez vos {n_crit} critères")
    defaults = DEFAULT_NAMES.get(n_crit, [f"Critère {i+1}" for i in range(n_crit)])
    crit_names = []
    cols_input = st.columns(2)
    for i in range(n_crit):
        key = f"name_c{i+1}"
        with cols_input[i % 2]:
            crit_names.append(st.text_input(
                f"Critère C{i+1}",
                value=st.session_state.get(key, defaults[i]),
                key=key
            ))

    st.markdown('<div class="gold-sep"></div>', unsafe_allow_html=True)
    st.markdown("#### Récapitulatif")
    rc = st.columns(min(n_crit, 4))
    for i, name in enumerate(crit_names):
        with rc[i % len(rc)]:
            st.markdown(f'<div class="metric-box"><b>C{i+1}</b> — {name}</div>',
                        unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    Matrice AHP : <b>{n_crit}×{n_crit}</b> &nbsp;·&nbsp;
    Comparaisons : <b>{n_crit*(n_crit-1)//2}</b> &nbsp;·&nbsp;
    RI Saaty (n={n_crit}) : <b>{RI_TABLE.get(n_crit, '?')}</b>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Matrice AHP
# ══════════════════════════════════════════════════════════════════════════════
elif step == "② Matrice AHP":
    n_crit = st.session_state.get("n_crit", 4)
    names  = [st.session_state.get(f"name_c{i+1}", f"C{i+1}") for i in range(n_crit)]
    PAIRS  = get_pairs(n_crit)

    st.subheader(f"ÉTAPE 2 — Matrice AHP {n_crit}×{n_crit} "
                 f"({n_crit*(n_crit-1)//2} comparaisons)")
    st.info("Saisissez les valeurs aᵢⱼ (entier ou fraction ex: 1/3). "
            "Les réciproques sont calculées automatiquement.")

    # Initialisation des valeurs en session
    for (i, j) in PAIRS:
        key = f"aij_{i}{j}"
        if key not in st.session_state:
            st.session_state[key] = str(round(DEFAULT_AHP_VALS.get((i,j), 1.0), 4))

    # Tableau de saisie
    st.markdown("#### 📝 Comparaisons par paires")
    hdr = st.columns([2, 1.8, 1.5, 1.8, 1.5, 3])
    for h, lbl in zip(hdr, ["Paire", "Critère i", "aᵢⱼ", "Critère j",
                              "Réciproque", "Interprétation"]):
        h.markdown(f"**{lbl}**")

    raw_vals     = {}
    parse_errors = []

    for k, (i, j) in enumerate(PAIRS):
        key  = f"aij_{i}{j}"
        cols = st.columns([2, 1.8, 1.5, 1.8, 1.5, 3])
        cols[0].markdown(f"Cmp. {k+1}")
        cols[1].markdown(f"**{names[i]}**")
        raw = cols[2].text_input(
            f"a{i+1}{j+1}", value=st.session_state[key],
            key=key, label_visibility="collapsed"
        )
        raw_vals[(i, j)] = raw
        try:
            val = (float(raw.split("/")[0]) / float(raw.split("/")[1])
                   if "/" in str(raw) else float(str(raw).replace(",", ".")))
            if val <= 0:
                raise ValueError
            cols[3].markdown(f"**{names[j]}**")
            cols[4].markdown(f"`{1/val:.4f}`")
            cols[5].markdown(f"*{interpret_saaty(val, names[i], names[j])}*")
        except Exception:
            parse_errors.append(k + 1)
            cols[3].markdown(f"**{names[j]}**")
            cols[4].markdown("⚠️")
            cols[5].markdown("❌ Valeur invalide")

    st.markdown('<div class="gold-sep"></div>', unsafe_allow_html=True)

    if parse_errors:
        st.error(f"Valeurs invalides aux comparaisons : {parse_errors}.")
    else:
        vals_list = []
        for (i, j) in PAIRS:
            r = str(raw_vals[(i, j)])
            vals_list.append(
                float(r.split("/")[0]) / float(r.split("/")[1])
                if "/" in r else float(r.replace(",", "."))
            )

        A = build_matrix(vals_list, n_crit)
        w, lam_max, CI, CR = ahp_weights(A)

        st.session_state.update({
            "ahp_weights": w, "ahp_lam": lam_max,
            "ahp_CI": CI, "ahp_CR": CR,
            "ahp_names": names, "ahp_matrix": A,
        })

        # Matrice
        st.markdown("#### Matrice de comparaisons complète")
        st.dataframe(
            pd.DataFrame(A.round(4), index=names, columns=names)
              .style.format("{:.4f}"),
            use_container_width=True
        )

        # Poids
        st.markdown("#### Poids AHP calculés")
        sorted_idx = np.argsort(-w)
        pw_cols = st.columns(n_crit)
        for rank, idx in enumerate(sorted_idx):
            with pw_cols[idx]:
                st.metric(f"C{idx+1} — {names[idx]}", f"{w[idx]*100:.1f} %",
                          delta=f"Rang {rank+1}")

        fig_w = px.bar(
            x=names, y=w * 100,
            labels={"x": "Critère", "y": "Poids (%)"},
            title="Distribution des poids AHP",
            color=w * 100, color_continuous_scale="Blues",
            text=[f"{wi*100:.1f}%" for wi in w],
        )
        fig_w.update_traces(textposition="outside")
        fig_w.update_layout(
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_range=[0, max(w) * 125],
        )
        st.plotly_chart(fig_w, use_container_width=True)

        # Cohérence
        st.markdown("#### Vérification de la cohérence (CR)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("λ_max", f"{lam_max:.4f}")
        mc2.metric("CI",    f"{CI:.4f}", help="(λ_max − n) / (n − 1)")
        mc3.metric("RI",    f"{RI_TABLE.get(n_crit, '?')}", help=f"n={n_crit}")
        mc4.metric("CR",    f"{CR*100:.2f} %",
                   delta="< 10% ✓" if CR < 0.10 else "> 10% ✗",
                   delta_color="normal" if CR < 0.10 else "inverse")

        badge = (f'<span class="cr-ok">✅ COHÉRENCE VALIDÉE — CR = {CR*100:.2f}% &lt; 10%</span>'
                 if CR < 0.10 else
                 f'<span class="cr-ko">❌ INCOHÉRENCE — CR = {CR*100:.2f}% &gt; 10% — '
                 f'Révisez vos jugements</span>')
        st.markdown(badge, unsafe_allow_html=True)

        if n_crit > 5 and CR >= 0.10:
            st.warning(f"Avec {n_crit} critères ({n_crit*(n_crit-1)//2} comparaisons), "
                       "maintenir CR < 10% est difficile. Envisagez de regrouper des critères.")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Import des données
# ══════════════════════════════════════════════════════════════════════════════
elif step == "③ Importer les données":
    n_crit = st.session_state.get("n_crit", 4)
    names  = st.session_state.get(
        "ahp_names",
        [st.session_state.get(f"name_c{i+1}", f"C{i+1}") for i in range(n_crit)]
    )
    crit_cols = [f"C{i+1}" for i in range(n_crit)]

    st.subheader("ÉTAPE 3 — Importer le fichier fournisseurs")
    cols_txt = ", ".join([f"**{n}**" for n in names])
    st.info(f"Le fichier doit contenir : **Fournisseur**, {cols_txt}")

    # Générateur exemple
    with st.expander(f"📥 Fichier exemple (500 fournisseurs, {n_crit} critères)"):
        np.random.seed(42)
        n_ex = 500
        n_s, n_e = int(n_ex * 0.17), int(n_ex * 0.32)
        n_b      = n_ex - n_s - n_e

        mus  = {"s": [14.5, 10.0] + [25, 8, 8, 7, 7, 7][:n_crit-2],
                "e": [12.5,  8.5] + [14, 5, 5, 5, 5, 5][:n_crit-2],
                "b": [10.5,  7.0] + [ 5, 3, 3, 3, 3, 3][:n_crit-2]}
        sigs = {"s": [0.6,  0.5] + [5, 2, 2, 2, 2, 2][:n_crit-2],
                "e": [0.7,  0.6] + [4, 2, 2, 2, 2, 2][:n_crit-2],
                "b": [0.8,  0.7] + [3, 2, 2, 2, 2, 2][:n_crit-2]}
        ns_map = {"s": n_s, "e": n_e, "b": n_b}

        arrs = {c: [] for c in range(n_crit)}
        for grp in ["s", "e", "b"]:
            n_g = ns_map[grp]
            for idx in range(n_crit):
                mu, sig = mus[grp][idx], sigs[grp][idx]
                col_data = (np.random.lognormal(mu, sig, n_g) if idx < 2
                            else np.clip(np.random.normal(mu, sig, n_g), 0, None))
                arrs[idx].append(col_data)

        shuf = np.random.permutation(n_ex)
        ex_dict = {"Fournisseur": [f"F{str(i+1).zfill(3)}" for i in range(n_ex)]}
        for i in range(n_crit):
            ex_dict[names[i]] = np.concatenate(arrs[i])[shuf].round(2)

        buf_ex = io.BytesIO()
        pd.DataFrame(ex_dict).to_excel(buf_ex, index=False)
        st.download_button("⬇️ Télécharger exemple.xlsx", data=buf_ex.getvalue(),
                           file_name="fournisseurs_exemple.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    uploaded = st.file_uploader("Importer votre fichier (.xlsx ou .csv)", type=["xlsx","csv"])
    if uploaded:
        try:
            df = (pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
                  else pd.read_excel(uploaded))
            st.success(f"✅ {len(df)} fournisseurs · {len(df.columns)} colonnes")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("#### Correspondance des colonnes")
            all_cols = list(df.columns)
            map_cols = st.columns(n_crit + 1)

            col_nom = map_cols[0].selectbox(
                "Fournisseur", all_cols,
                index=all_cols.index("Fournisseur") if "Fournisseur" in all_cols else 0
            )
            col_crit = [
                map_cols[i+1].selectbox(names[i], all_cols, index=min(i+1, len(all_cols)-1))
                for i in range(n_crit)
            ]

            if st.button("✅ Valider et enregistrer", type="primary"):
                df_clean = df[[col_nom] + col_crit].copy()
                df_clean.columns = ["Fournisseur"] + crit_cols
                for c in crit_cols:
                    df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
                df_clean = df_clean.dropna()
                st.session_state["df_fournisseurs"] = df_clean
                st.success(f"✅ {len(df_clean)} fournisseurs enregistrés. "
                           "Passez à l'étape ④.")
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")
    else:
        st.warning("Aucun fichier importé. Chargez un fichier pour continuer.")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Classification & Résultats
# ══════════════════════════════════════════════════════════════════════════════
elif step == "④ Résultats & Tiers":
    n_crit    = st.session_state.get("n_crit", 4)
    crit_cols = [f"C{i+1}" for i in range(n_crit)]
    z_cols    = [f"Z_C{i+1}" for i in range(n_crit)]

    st.subheader(f"ÉTAPE 4 — Classification AHP + K-means ({n_crit} critères)")

    ok_weights = "ahp_weights" in st.session_state
    ok_data    = "df_fournisseurs" in st.session_state
    ok_cr      = st.session_state.get("ahp_CR", 1.0) <= 0.10

    c1, c2, c3 = st.columns(3)
    c1.metric("Poids AHP", "✅ Calculés" if ok_weights else "❌ Manquants")
    c2.metric("CR", f"{st.session_state.get('ahp_CR',1)*100:.2f}%" if ok_weights else "—",
              delta="Validé" if ok_cr else "À corriger",
              delta_color="normal" if ok_cr else "inverse")
    c3.metric("Données",
              f"{len(st.session_state['df_fournisseurs'])} fournisseurs"
              if ok_data else "❌ Manquantes")

    st.markdown('<div class="gold-sep"></div>', unsafe_allow_html=True)

    if not ok_weights:
        st.error("Calculez d'abord les poids AHP à l'étape ②.")
        st.stop()
    if not ok_data:
        st.error("Importez les données fournisseurs à l'étape ③.")
        st.stop()
    if not ok_cr:
        st.warning("CR > 10% : résultats affichés mais jugements à réviser (étape ②).")

    w     = st.session_state["ahp_weights"]
    names = st.session_state.get(
        "ahp_names",
        [st.session_state.get(f"name_c{i+1}", f"C{i+1}") for i in range(n_crit)]
    )
    df  = st.session_state["df_fournisseurs"].copy()
    p_t1    = st.session_state.get("km_p_tier1", 80)
    p_t3    = st.session_state.get("km_p_tier3", 45)
    method  = st.session_state.get("seg_method", "Seuils directs (recommandé)")

    with st.expander("ℹ️ Pourquoi la segmentation opère sur le score AHP 1D"):
        st.markdown(f"""
**Pourquoi PAS K-means en espace n-D ?**

Après standardisation, chaque critère suit N(0,1). Le score AHP est une combinaison
linéaire de n variables N(0,1) → il suit lui aussi une loi quasi-normale symétrique.
K-means sur une loi symétrique **converge toujours vers ~33%/33%/33%**,
quelle que soit l'initialisation. C'est mathématiquement correct mais méthodologiquement
faux pour la segmentation fournisseurs.

**La bonne approche :**

AHP a déjà fait le travail de synthèse — le **score 1D** `S = Z · w` est la
représentation la plus fidèle de la performance globale d'un fournisseur.
La segmentation doit opérer sur ce score :

| Méthode | Principe | Garantie Pareto |
|---|---|---|
| **Seuils directs** | Score ≥ {p_t1}e pct → T1, < {p_t3}e pct → T3 | ✅ Exacte |
| **K-means 1D** | Natural breaks sur le score 1D | ✅ Approx. (données structurées) |
        """)

    if st.button("🚀 Lancer la classification", type="primary"):

        # ── Z-scores ─────────────────────────────────────────────────────────
        X = df[crit_cols].values.astype(float)
        mu_x, sig_x = X.mean(axis=0), X.std(axis=0, ddof=0)
        sig_x[sig_x == 0] = 1
        Z      = (X - mu_x) / sig_x
        scores = Z @ w   # score AHP 1D

        # ── Segmentation ─────────────────────────────────────────────────────
        if method == "Seuils directs (recommandé)":
            labels, thr1, thr3, cluster_means = segment_direct(scores, p_t1, p_t3)
            method_label = f"Seuils directs (≥{thr1:.3f} → T1, <{thr3:.3f} → T3)"
        else:
            labels, thr1, thr3, cluster_means = segment_kmeans1d(scores, p_t1, p_t3)
            method_label = f"K-means 1D (ruptures naturelles : {thr1:.3f} | {thr3:.3f})"

        # DataFrame résultat
        tier_map   = {0: "Tier 1", 1: "Tier 2", 2: "Tier 3"}
        profil_map = {0: "Stratégique", 1: "Essentiel", 2: "Basique"}
        for i in range(n_crit):
            df[f"Z_C{i+1}"] = Z[:, i].round(4)
        df["Score AHP"] = scores.round(4)
        df["Tier"]      = [tier_map[l]   for l in labels]
        df["Profil"]    = [profil_map[l] for l in labels]

        st.session_state.update({
            "df_results":    df,
            "seg_centroids": cluster_means,
            "seg_thr1":      thr1,
            "seg_thr3":      thr3,
            "seg_method_lbl": method_label,
        })

        cnt = df["Tier"].value_counts()
        t1, t2, t3 = cnt.get("Tier 1", 0), cnt.get("Tier 2", 0), cnt.get("Tier 3", 0)
        N  = len(df)

        # KPIs résumé
        st.markdown("### Résumé de la segmentation")
        st.caption(f"Méthode : {method_label}")
        k1, k2, k3 = st.columns(3)
        k1.markdown(f'<div class="tier-card tier1">Tier 1<br>Stratégiques<br>'
                    f'{t1} ({t1/N*100:.0f}%)</div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="tier-card tier2">Tier 2<br>Essentiels<br>'
                    f'{t2} ({t2/N*100:.0f}%)</div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="tier-card tier3">Tier 3<br>Basiques<br>'
                    f'{t3} ({t3/N*100:.0f}%)</div>', unsafe_allow_html=True)

        sm = st.columns(3)
        sm[0].metric("Score moyen Tier 1", f"{cluster_means[0]:+.4f}")
        sm[1].metric("Score moyen Tier 2", f"{cluster_means[1]:+.4f}")
        sm[2].metric("Score moyen Tier 3", f"{cluster_means[2]:+.4f}")

        st.markdown('<div class="gold-sep"></div>', unsafe_allow_html=True)

        # ── Visualisations ────────────────────────────────────────────────────
        st.markdown("### Visualisations")
        v1, v2 = st.columns(2)
        with v1:
            fig_hist = px.histogram(
                df, x="Score AHP", color="Tier",
                color_discrete_map={"Tier 1":"#1e7e34","Tier 2":"#856404","Tier 3":"#721c24"},
                nbins=40, barmode="overlay", opacity=0.75,
                title="Distribution des scores AHP par Tier",
            )
            # Seuils de coupure
            fig_hist.add_vline(x=thr1, line_dash="dash", line_color="#1e7e34",
                               annotation_text=f"Seuil T1={thr1:.3f}")
            fig_hist.add_vline(x=thr3, line_dash="dash", line_color="#dc3545",
                               annotation_text=f"Seuil T3={thr3:.3f}")
            fig_hist.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_hist, use_container_width=True)

        with v2:
            fig_pie = px.pie(
                df, names="Tier", color="Tier",
                color_discrete_map=TIER_COLORS,
                title="Répartition des fournisseurs", hole=0.4,
            )
            fig_pie.update_traces(textinfo="label+percent+value")
            fig_pie.update_layout(paper_bgcolor="white")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Scatter C1 vs Score AHP
        fig_sc = px.scatter(
            df, x="C1", y="Score AHP", color="Tier",
            hover_data=["Fournisseur"] + crit_cols[1:],
            color_discrete_map=TIER_COLORS,
            title=f"Score AHP vs {names[0]}",
            labels={"C1": names[0]},
        )
        fig_sc.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_sc, use_container_width=True)

        # Radar profil moyen par tier
        radar_data = df.groupby("Tier")[z_cols].mean()
        fig_radar  = go.Figure()
        for tier in ["Tier 1","Tier 2","Tier 3"]:
            if tier in radar_data.index:
                row  = radar_data.loc[tier]
                rv   = list(row) + [row.iloc[0]]
                cats = names + [names[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=rv, theta=cats, name=tier, fill="toself",
                    line_color=TIER_COLORS[tier],
                ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title=f"Profil moyen (z-scores) par Tier — {n_crit} critères",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('<div class="gold-sep"></div>', unsafe_allow_html=True)

        # ── Table détaillée ───────────────────────────────────────────────────
        st.markdown("### Détail des fournisseurs")
        tier_filter = st.multiselect("Filtrer par tier",
            ["Tier 1","Tier 2","Tier 3"], default=["Tier 1","Tier 2","Tier 3"])
        df_show = (df[df["Tier"].isin(tier_filter)]
                   .sort_values("Score AHP", ascending=False))

        rename_map = {f"C{i+1}": names[i] for i in range(n_crit)}
        fmt_map    = {"Score AHP": "{:.4f}"}
        for c in z_cols:
            fmt_map[c] = "{:.4f}"

        st.dataframe(
            df_show.rename(columns=rename_map)
                   .style.applymap(
                       lambda v: f"background-color: {TIER_BG.get(v,'white')}",
                       subset=["Tier"]
                   ).format(fmt_map),
            use_container_width=True, height=400,
        )

        # ── Export Excel ──────────────────────────────────────────────────────
        st.markdown("#### 📤 Export des résultats")
        buf_xl = io.BytesIO()
        with pd.ExcelWriter(buf_xl, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Tous", index=False)
            df[df["Tier"]=="Tier 1"].to_excel(writer, sheet_name="Tier 1 - Stratégiques", index=False)
            df[df["Tier"]=="Tier 2"].to_excel(writer, sheet_name="Tier 2 - Essentiels",   index=False)
            df[df["Tier"]=="Tier 3"].to_excel(writer, sheet_name="Tier 3 - Basiques",      index=False)

            pd.DataFrame({
                "Critère": names,
                "Poids":   w,
                "%":       [f"{wi*100:.1f}%" for wi in w],
                "Rang":    (np.argsort(-w)+1)[np.argsort(np.argsort(-w))].tolist(),
            }).to_excel(writer, sheet_name="Poids AHP", index=False)

            pd.DataFrame({
                "Paramètre": ["n critères","λ_max","CI","RI","CR","Seuil CR",
                               "Décision","Méthode segmentation",
                               "Seuil Tier 1 (percentile)","Seuil Tier 3 (percentile)",
                               "Score seuil Tier 1","Score seuil Tier 3"],
                "Valeur": [
                    str(n_crit),
                    f"{st.session_state['ahp_lam']:.4f}",
                    f"{st.session_state['ahp_CI']:.4f}",
                    f"{RI_TABLE.get(n_crit,'?')}",
                    f"{st.session_state['ahp_CR']:.4f}",
                    "0.10",
                    "Validé" if ok_cr else "Invalide",
                    method,
                    f"{p_t1}e pct. (top {100-p_t1}%)",
                    f"{p_t3}e pct. (bottom {p_t3}%)",
                    f"{thr1:.4f}",
                    f"{thr3:.4f}",
                ],
            }).to_excel(writer, sheet_name="Synthèse AHP", index=False)

        st.download_button(
            "⬇️ Télécharger les résultats (.xlsx)",
            data=buf_xl.getvalue(),
            file_name="segmentation_AHP_Kmeans_v2.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    elif "df_results" in st.session_state:
        st.info("Résultats de la dernière classification disponibles. "
                "Cliquez sur 'Lancer la classification' pour recalculer.")
