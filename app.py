

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from abm.simulation import ElectionModel, run_comparative

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="Electoral Systems ABM", layout="wide")



# Data loading

@st.cache_data
def load_data():
    voters = pd.read_csv(DATA_DIR / "processed_voters.csv")
    parties_uk = pd.read_csv(DATA_DIR / "processed_parties_uk.csv")
    parties_all = pd.read_csv(DATA_DIR / "processed_parties_all.csv")
    validation = pd.read_csv(DATA_DIR / "uk_ge2024_validation.csv")
    validation_summary = pd.read_csv(DATA_DIR / "uk_ge2024_validation_summary.csv")
    return voters, parties_uk, parties_all, validation, validation_summary


try:
    voters_df, parties_uk_df, parties_all_df, validation_df, validation_summary_df = load_data()
except FileNotFoundError:
    st.error(
        "Processed data files not found. Please run **`python data_prep.py`** "
        "from the project folder first (see README) before launching the app."
    )
    st.stop()



# Sidebar controls

st.sidebar.title("Simulation Settings")

st.sidebar.subheader("Party seeding (from CHES)")
countries = sorted(parties_all_df["country_name"].dropna().unique())
default_country_idx = countries.index("United Kingdom") if "United Kingdom" in countries else 0
country = st.sidebar.selectbox("Country (real party positions)", countries,
                                index=default_country_idx)
years_for_country = sorted(parties_all_df.loc[
    parties_all_df["country_name"] == country, "year"].unique(), reverse=True)
year = st.sidebar.selectbox("Election year", years_for_country, index=0)

parties_df = parties_all_df[(parties_all_df["country_name"] == country) &
                             (parties_all_df["year"] == year)].copy()
parties_df = parties_df.sort_values("vote", ascending=False).head(8).reset_index(drop=True)

st.sidebar.subheader("Population & institutions")
n_voters = st.sidebar.slider("Number of voter agents", 200, 3000, 1200, step=100)
n_districts = st.sidebar.slider("Number of geographic districts", 5, 100, 20, step=5)
total_seats = st.sidebar.slider("Total seats in legislature", 20, 300, 100, step=10)
pr_threshold = st.sidebar.slider("PR vote threshold to win seats", 0.0, 0.10, 0.05, step=0.01)

st.sidebar.subheader("Voter opinion dynamics")
confidence_eps = st.sidebar.slider("Bounded-confidence threshold (ε)", 0.1, 1.0, 0.45, step=0.05,
                                    help="Voters only move toward neighbours whose ideology is "
                                         "within this distance (Hegselmann–Krause model).")
susceptibility = st.sidebar.slider("Voter susceptibility to influence", 0.0, 0.5, 0.15, step=0.01)
network_k = st.sidebar.slider("Social network connections per voter (k)", 2, 20, 6, step=1)
network_p = st.sidebar.slider("Network rewiring probability (p)", 0.0, 1.0, 0.1, step=0.05,
                               help="0 = regular ring lattice, 1 = fully random network "
                                    "(Watts–Strogatz small-world model).")

st.sidebar.subheader("Party strategy")
party_adapt_rate = st.sidebar.slider("Party adaptation rate", 0.0, 0.3, 0.08, step=0.01,
                                      help="How strongly parties move toward their supporter "
                                           "base's centroid each campaign cycle.")
n_cycles = st.sidebar.slider("Campaign cycles before election", 0, 20, 6, step=1)

seed = st.sidebar.number_input("Random seed", value=42, step=1)



model_kwargs = dict(
    n_voters=n_voters, n_districts=n_districts, total_seats=total_seats,
    network_k=network_k, network_p=network_p, confidence_eps=confidence_eps,
    susceptibility=susceptibility, party_adapt_rate=party_adapt_rate,
    pr_threshold=pr_threshold, seed=int(seed),
)

if len(parties_df) < 2:
    st.error("Need at least 2 parties for the selected country/year - pick another combination.")
    st.stop()



st.title("Agent-Based Modelling of Electoral Systems")
st.markdown(
    "Simulating **voter dynamics**, **party strategy**, and **democratic outcomes** "
    "under First-Past-the-Post, Proportional Representation, and Mixed-Member "
    "Proportional systems - grounded in real CCES, CHES, and UK GE2024 data."
)

tab_sim, tab_compare, tab_validate = st.tabs(
    ["Simulate", "Compare Systems", "Real-World Validation"]
)



# Helper plotting functions

def ideology_scatter(voters, parties, title):
    vx = [v.econ_ideology for v in voters]
    vy = [v.social_ideology for v in voters]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=vx, y=vy, mode="markers",
        marker=dict(size=5, color="rgba(120,120,180,0.35)"),
        name="Voters"))
    for p in parties:
        fig.add_trace(go.Scatter(
            x=[p.econ_ideology], y=[p.social_ideology], mode="markers+text",
            marker=dict(size=18, color=p.color, line=dict(width=2, color="black")),
            text=[p.name], textposition="top center", name=p.name))
    fig.update_layout(title=title, xaxis_title="Economic left (-1) — right (+1)",
                       yaxis_title="Social liberal/GAL (-1) — traditional/TAN (+1)",
                       height=500, xaxis_range=[-1.1, 1.1], yaxis_range=[-1.1, 1.1])
    return fig


def party_trajectory_plot(model):
    fig = go.Figure()
    for p in model.parties:
        xs = [pos[0] for pos in p.history]
        ys = [pos[1] for pos in p.history]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=p.name,
                                  line=dict(color=p.color), marker=dict(size=6)))
    fig.update_layout(title="Party ideological trajectories across campaign cycles",
                       xaxis_title="Economic axis", yaxis_title="Social axis",
                       height=450, xaxis_range=[-1.1, 1.1], yaxis_range=[-1.1, 1.1])
    return fig


def polarisation_trend_plot(model):
    hist = pd.DataFrame(model.history)
    fig = px.line(hist, x="cycle", y="polarisation", markers=True,
                  title="Voter ideological polarisation over campaign cycles")
    fig.update_layout(height=350, xaxis_title="Campaign cycle",
                       yaxis_title="Polarisation (std. dev. of ideology)")
    return fig


def vote_seat_bar(table, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["party"], y=table["vote_share"] * 100,
                          name="Vote share %", marker_color="#8ecae6"))
    fig.add_trace(go.Bar(x=table["party"], y=table["seat_share"] * 100,
                          name="Seat share %", marker_color="#023047"))
    fig.update_layout(barmode="group", title=title, height=420,
                       yaxis_title="Share (%)")
    return fig


def metric_row(summary):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Turnout", f"{summary['turnout']*100:.1f}%")
    c2.metric("Gallagher Index", f"{summary['gallagher_index']:.2f}",
              help="Disproportionality between votes and seats. Lower = fairer.")
    c3.metric("Eff. Parties (seats)", f"{summary['enp_seats']:.2f}")
    c4.metric("Polarisation", f"{summary['polarisation']:.3f}")
    c5.metric("Ideological diversity", f"{summary['ideological_diversity']:.3f}")



# Simulate

with tab_sim:
    system = st.radio("Electoral system", ["FPTP", "PR", "MMP"], horizontal=True)
    run_btn = st.button("Run simulation", type="primary")

    if run_btn:
        with st.spinner("Building agents, running campaign cycles, holding the election..."):
            model = ElectionModel(voters_df, parties_df, **model_kwargs)
            init_snapshot = copy.deepcopy(model)
            model.run_cycles(n_cycles)
            summary, table = model.hold_election(system=system)
        st.session_state["sim_model"] = model
        st.session_state["sim_init"] = init_snapshot
        st.session_state["sim_summary"] = summary
        st.session_state["sim_table"] = table
        st.session_state["sim_system"] = system

    if "sim_model" in st.session_state:
        model = st.session_state["sim_model"]
        init_model = st.session_state["sim_init"]
        summary = st.session_state["sim_summary"]
        table = st.session_state["sim_table"]

        st.subheader(f"Results — {st.session_state['sim_system']}")
        metric_row(summary)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ideology_scatter(init_model.voters, init_model.parties,
                                              "Voter/party ideology — before campaign"),
                             use_container_width=True)
        with c2:
            st.plotly_chart(ideology_scatter(model.voters, model.parties,
                                              "Voter/party ideology — after campaign"),
                             use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(party_trajectory_plot(model), use_container_width=True)
        with c4:
            st.plotly_chart(polarisation_trend_plot(model), use_container_width=True)

        st.plotly_chart(vote_seat_bar(table, "Vote share vs. seat share by party"),
                         use_container_width=True)
        st.dataframe(table.drop(columns=["color"]).style.format({
            "vote_share": "{:.1%}", "seat_share": "{:.1%}",
            "econ_ideology": "{:.2f}", "social_ideology": "{:.2f}",
            "real_vote_pct": "{:.1f}", "real_seat_pct": "{:.1f}",
        }), use_container_width=True)
    else:
        st.info("Configure parameters in the sidebar, then click **Run simulation**.")



# Compare systems

with tab_compare:
    st.markdown("Runs the **same** voter population, social network and campaign "
                "dynamics through FPTP, PR, and MMP, so any differences in outcome "
                "come only from the electoral rule itself.")
    compare_btn = st.button("Run comparative simulation", type="primary")

    if compare_btn:
        with st.spinner("Running FPTP, PR and MMP on the same population..."):
            base_model, summary_df, tables = run_comparative(
                voters_df, parties_df, n_cycles=n_cycles, systems=("FPTP", "PR", "MMP"),
                **model_kwargs)
        st.session_state["cmp_summary"] = summary_df
        st.session_state["cmp_tables"] = tables

    if "cmp_summary" in st.session_state:
        summary_df = st.session_state["cmp_summary"]
        tables = st.session_state["cmp_tables"]

        m1 = px.bar(summary_df, x="system", y="gallagher_index", color="system",
                     title="Representational disproportionality (Gallagher Index) by system",
                     text_auto=".2f")
        m2 = px.bar(summary_df, x="system", y="enp_seats", color="system",
                     title="Effective number of parties (seats) by system", text_auto=".2f")
        c1, c2 = st.columns(2)
        c1.plotly_chart(m1, use_container_width=True)
        c2.plotly_chart(m2, use_container_width=True)

        st.subheader("Seat share by party, across systems")
        seat_compare = pd.concat([
            t.assign(system=sys_name)[["system", "party", "seat_share", "vote_share"]]
            for sys_name, t in tables.items()
        ])
        fig = px.bar(seat_compare, x="party", y="seat_share", color="system",
                     barmode="group", title="Seat share by party across FPTP / PR / MMP")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full comparative metrics")
        st.dataframe(summary_df.set_index("system").style.format({
            "turnout": "{:.1%}", "gallagher_index": "{:.2f}", "enp_votes": "{:.2f}",
            "enp_seats": "{:.2f}", "polarisation": "{:.3f}", "ideological_diversity": "{:.3f}",
        }), use_container_width=True)
    else:
        st.info("Click **Run comparative simulation** to compare all three systems.")



# Real-world validation

with tab_validate:
    st.markdown(
        "Benchmarks the simulation against **real UK General Election 2024** data: "
        "actual party vote/seat shares (CHES) and actual seat swings among defeated MPs."
    )

    uk_only = country == "United Kingdom" and year == 2024
    if not uk_only:
        st.warning("Select **United Kingdom, 2024** in the sidebar to enable a direct "
                   "real-vs-simulated comparison (validation data only exists for that "
                   "election).")

    val_btn = st.button("Run FPTP simulation for validation", type="primary",
                         disabled=not uk_only)

    if val_btn:
        with st.spinner("Simulating UK 2024 under FPTP..."):
            vmodel = ElectionModel(voters_df, parties_df, **model_kwargs)
            vmodel.run_cycles(n_cycles)
            vsummary, vtable = vmodel.hold_election(system="FPTP")
        st.session_state["val_table"] = vtable

    if "val_table" in st.session_state:
        vtable = st.session_state["val_table"]
        comp = vtable[["party", "vote_share", "seat_share", "real_vote_pct", "real_seat_pct"]].copy()
        comp["sim_vote_pct"] = comp["vote_share"] * 100
        comp["sim_seat_pct"] = comp["seat_share"] * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(x=comp["party"], y=comp["real_seat_pct"], name="Real seat % (CHES 2024)",
                              marker_color="#264653"))
        fig.add_trace(go.Bar(x=comp["party"], y=comp["sim_seat_pct"], name="Simulated seat % (FPTP)",
                              marker_color="#e76f51"))
        fig.update_layout(barmode="group", height=420,
                           title="Real vs. simulated seat share — UK 2024")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comp[["party", "real_vote_pct", "sim_vote_pct", "real_seat_pct", "sim_seat_pct"]]
                     .style.format("{:.1f}", subset=["real_vote_pct", "sim_vote_pct",
                                                       "real_seat_pct", "sim_seat_pct"]),
                     use_container_width=True)
        st.caption(
            "Note: the simulation is a small, stylised agent population (hundreds-thousands "
            "of agents vs ~48 million real UK electors) so it is not expected to exactly "
            "reproduce 2024 results. The comparison is used descriptively - e.g. to check "
            "the model reproduces the *direction* of FPTP's well-known bias toward "
            "manufacturing majorities for the largest party at the expense of smaller "
            "parties, which is exactly what the real 2024 Labour/Reform UK result shows."
        )

    st.divider()
    st.subheader("Real seat swings among defeated MPs — UK GE2024")
    st.markdown(
        "This is a **partial** ground-truth dataset (only constituencies where the "
        "sitting MP lost their seat), used descriptively rather than as a full election result."
    )
    fig2 = px.bar(validation_summary_df, x="winning_party", y="seats_gained_from_defeated_mps",
                  title="Seats gained (from a defeated incumbent) by party, GE2024",
                  color="winning_party")
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("View raw defeated-MP records"):
        st.dataframe(validation_df, use_container_width=True)



