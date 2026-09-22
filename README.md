# Agent-Based Modelling of Electoral Systems

A Python agent-based simulation of elections — voters, adaptive parties,
opinion dynamics and three electoral systems (FPTP, PR, MMP) — with an
interactive Streamlit dashboard, grounded in real CCES, CHES, and UK
GE2024 data.

## Project structure

```
election_abm/
├── app.py                     # Streamlit GUI (run this)
├── data_prep.py                # (optional) regenerates processed data from raw files
├── requirements.txt
├── abm/
│   ├── agents.py               # VoterAgent, PartyAgent
│   ├── electoral_systems.py    # FPTP / PR (D'Hondt) / MMP (Sainte-Laguë)
│   ├── metrics.py               # Gallagher index, ENP, polarisation, etc.
│   └── simulation.py            # ElectionModel (Mesa Model) + run_comparative()
└── data/
    ├── processed_voters.csv           # from CCES 2022 (pre-processed, ready to use)
    ├── processed_parties_uk.csv       # UK 2024 parties from CHES
    ├── processed_parties_all.csv      # all CHES countries/years
    ├── uk_ge2024_validation.csv       # defeated-MP records, GE2024
    └── uk_ge2024_validation_summary.csv
```

The processed data files are already included, so **you do not need the
original 190MB CCES file to run the app** — everything needed ships in `/data`.

## 1. Setup

Requires Python 3.10+.

```bash
# from inside the election_abm/ folder
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Run the app

```bash
streamlit run app.py
```

This opens the dashboard in your browser (usually `http://localhost:8501`).

## 3. Using the dashboard

1. **Sidebar** — choose the country/year to seed real party positions from
   CHES (defaults to United Kingdom, 2024), population size, number of
   districts/seats, opinion-dynamics parameters (bounded-confidence ε,
   network structure), and party adaptation rate.
2. **▶️ Simulate tab** — pick an electoral system and click *Run simulation*
   to see voter/party ideology before & after the campaign, party
   trajectories, a polarisation trend, and the vote-share vs seat-share
   outcome.
3. **⚖️ Compare Systems tab** — runs FPTP, PR and MMP on the *same* starting
   population so you can directly compare disproportionality (Gallagher
   Index), party fragmentation (effective number of parties), and seat
   shares across institutions.
4. **✅ Real-World Validation tab** — (select United Kingdom / 2024 in the
   sidebar) compares the simulated FPTP outcome against the real 2024 UK
   General Election vote/seat shares (from CHES) and a chart of real seat
   swings among defeated MPs.
5. **ℹ️ About & Data tab** — full methodology and data-source documentation.

## 4. (Optional) Regenerating the processed data

Only needed if you want to rebuild `/data` yourself (e.g. a larger voter
sample, or refreshed source files). Place the three original raw files in a
`raw_data/` folder next to `data_prep.py`, using these exact filenames:

```
raw_data/CCES22_Common_OUTPUT_vv_topost.csv
raw_data/1999-2024_CHES_dataset_meansV2.csv
raw_data/Defeated-MPs-_Friday-1720_.csv
```

Then run:

```bash
python data_prep.py
```

This overwrites the CSVs in `/data`.

## 5. Using the ABM without the GUI (scripting / notebooks)

```python
import pandas as pd
from abm.simulation import ElectionModel, run_comparative

voters = pd.read_csv("data/processed_voters.csv")
parties = pd.read_csv("data/processed_parties_uk.csv")

# single system
model = ElectionModel(voters, parties, n_voters=1500, n_districts=20,
                       total_seats=100, seed=42)
model.run_cycles(6)                       # campaign / opinion-dynamics cycles
summary, results_table = model.hold_election(system="FPTP")
print(summary)
print(results_table)

# compare all three systems on the same population
base_model, summary_df, tables = run_comparative(
    voters, parties, n_cycles=6, n_voters=1500, n_districts=20,
    total_seats=100, seed=42)
print(summary_df)
```

## Methodology (short version)

- **Voters**: bootstrap-sampled from CCES 2022 micro-data; 2-D ideology
  (economic left–right; social liberal/GAL ↔ traditional/TAN proxy), placed
  on a Watts–Strogatz small-world social network.
- **Opinion dynamics**: bounded-confidence model (Hegselmann & Krause,
  2002) — voters shift toward network neighbours within a confidence
  threshold ε each campaign cycle; stronger partisans are more resistant.
- **Parties**: seeded with real CHES ideological coordinates; adapt
  partway toward their supporter-base centroid each cycle (Kollman,
  Miller & Page, 1992 style adaptive strategy).
- **Elections**: spatial (nearest-party) voting + valence + partisan-lean
  bonus; individual turnout draws from empirically-derived propensities;
  seats allocated by FPTP / PR (D'Hondt) / MMP (Sainte-Laguë top-up).
- **Metrics**: turnout, Gallagher disproportionality index, effective
  number of parties (Laakso & Taagepera, 1979), polarisation (std. dev. of
  ideology), seat-weighted ideological diversity.

See the **About & Data** tab in the app for full details, data-source
documentation, and stated limitations.
