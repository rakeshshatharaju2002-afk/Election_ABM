

import copy
import numpy as np
import pandas as pd
import networkx as nx
from mesa import Model

from .agents import VoterAgent, PartyAgent
from .electoral_systems import SYSTEMS
from . import metrics as M

DEFAULT_PARTY_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


class ElectionModel(Model):
    def __init__(self, voters_df, parties_df, n_voters=1500, n_districts=20,
                 total_seats=100, network_k=6, network_p=0.1,
                 confidence_eps=0.45, susceptibility=0.15,
                 party_adapt_rate=0.08, pr_threshold=0.05, seed=None):
        super().__init__(rng=seed)
        self.rng = np.random.default_rng(seed)
        self.n_voters = n_voters
        self.n_districts = n_districts
        self.total_seats = total_seats
        self.pr_threshold = pr_threshold
        self.cycle = 0
        self.history = []

        self._build_voters(voters_df, n_voters, confidence_eps, susceptibility)
        self._build_network(network_k, network_p)
        self._build_parties(parties_df, party_adapt_rate)


    def _build_voters(self, voters_df, n_voters, confidence_eps, susceptibility):
        sample = voters_df.sample(n=n_voters, replace=True,
                                   random_state=self.rng.integers(0, 1_000_000))
        self.voters = []
        for _, row in sample.iterrows():
            v = VoterAgent(
                self,
                econ_ideology=row["econ_ideology"],
                social_ideology=row["social_ideology"],
                partisan_lean=row["partisan_lean"],
                partisan_strength=row["partisan_strength"],
                turnout_propensity=row["turnout_propensity"],
                confidence_eps=confidence_eps,
                susceptibility=susceptibility,
            )
            self.voters.append(v)

    def _build_network(self, k, p):
        k = max(2, min(k, self.n_voters - 1))
        if k % 2 == 1:
            k += 1
        self.network = nx.watts_strogatz_graph(self.n_voters, k, p,
                                                 seed=int(self.rng.integers(0, 1_000_000)))
        self._idx_of = {v.unique_id: i for i, v in enumerate(self.voters)}

    def _build_parties(self, parties_df, adapt_rate):
        self.parties = []
        for i, (_, row) in enumerate(parties_df.iterrows()):
            valence = self.rng.normal(0, 0.05)
            p = PartyAgent(
                self,
                name=row["party"],
                econ_ideology=row["econ_ideology"],
                social_ideology=row["social_ideology"],
                adapt_rate=adapt_rate,
                valence=valence,
            )
            p.color = DEFAULT_PARTY_COLORS[i % len(DEFAULT_PARTY_COLORS)]
            p.real_vote_pct = row.get("vote", np.nan)
            p.real_seat_pct = row.get("seat", np.nan)
            p.history.append(tuple(p.position))
            self.parties.append(p)


    def _opinion_step(self):
        positions = {v.unique_id: v.position for v in self.voters}
        new_positions = {}
        for v in self.voters:
            i = self._idx_of[v.unique_id]
            neighbor_ids = [self.voters[j].unique_id for j in self.network.neighbors(i)]
            neighbor_pos = [positions[nid] for nid in neighbor_ids]
            v.update_opinion(neighbor_pos)
            new_positions[v.unique_id] = v.position

    def _party_step(self):

        for p in self.parties:
            supporters = []
            for v in self.voters:
                dists = [np.linalg.norm(v.position - pp.position) for pp in self.parties]
                if int(np.argmin(dists)) == self.parties.index(p):
                    supporters.append(v.position)
            p.adapt_position(supporters)

    def run_cycle(self):

        self._opinion_step()
        self._party_step()
        self.cycle += 1
        self.history.append({
            "cycle": self.cycle,
            "polarisation": M.polarisation_index(self.voters),
            "party_positions": {p.name: tuple(p.position) for p in self.parties},
        })

    def run_cycles(self, n):
        for _ in range(n):
            self.run_cycle()


    def hold_election(self, system="FPTP"):


        for p in self.parties:
            p.vote_count = 0
            p.seats = 0

        for v in self.voters:
            chosen = v.choose_party(self.parties)
            v.decide_turnout()
            if not v.voted:
                v.party_choice = None

        for v in self.voters:
            if v.voted and v.party_choice is not None:
                for p in self.parties:
                    if p.unique_id == v.party_choice:
                        p.vote_count += 1
                        break

        fn = SYSTEMS[system]
        kwargs = dict(n_districts=self.n_districts, total_seats=self.total_seats,
                      threshold=self.pr_threshold, rng=self.rng)
        seats = fn(self.voters, self.parties, **{k: v for k, v in kwargs.items()
                                                  if k in fn.__code__.co_varnames})
        for p in self.parties:
            p.seats = seats.get(p.name, 0)

        summary = M.summarise_election(self.voters, self.parties)
        summary["system"] = system
        summary["cycle"] = self.cycle
        results_table = pd.DataFrame([{
            "party": p.name,
            "vote_count": p.vote_count,
            "vote_share": p.vote_share,
            "seats": p.seats,
            "seat_share": p.seat_share,
            "econ_ideology": p.econ_ideology,
            "social_ideology": p.social_ideology,
            "real_vote_pct": getattr(p, "real_vote_pct", np.nan),
            "real_seat_pct": getattr(p, "real_seat_pct", np.nan),
            "color": p.color,
        } for p in self.parties]).sort_values("vote_share", ascending=False)

        return summary, results_table


def run_comparative(voters_df, parties_df, n_cycles=5, systems=("FPTP", "PR", "MMP"),
                     **model_kwargs):

    seed = model_kwargs.pop("seed", None) or np.random.default_rng().integers(0, 1_000_000)
    base_model = ElectionModel(voters_df, parties_df, seed=seed, **model_kwargs)
    base_model.run_cycles(n_cycles)

    all_summaries = []
    all_tables = {}
    for system in systems:
        model_copy = copy.deepcopy(base_model)
        summary, table = model_copy.hold_election(system=system)
        all_summaries.append(summary)
        all_tables[system] = table
    return base_model, pd.DataFrame(all_summaries), all_tables
