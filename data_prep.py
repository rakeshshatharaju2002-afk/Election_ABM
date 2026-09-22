

import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw_data"
OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

CCES_FILE = RAW_DIR / "CCES22_Common_OUTPUT_vv_topost.csv"
CHES_FILE = RAW_DIR / "1999-2024_CHES_dataset_meansV2.csv"
MPS_FILE = RAW_DIR / "Defeated-MPs-_Friday-1720_.csv"

N_VOTER_SAMPLE = 6000
RNG_SEED = 42



# processed_voters.csv

def process_cces():
    print("Processing CCES voter micro-data ...")
    cols = ["caseid", "pid7", "ideo5", "birthyr", "gender4", "educ", "race",
            "faminc_new", "inputstate", "urbancity", "votereg", "commonweight"]
    df = pd.read_csv(CCES_FILE, usecols=cols)


    df = df[(df["ideo5"].between(1, 5)) & (df["pid7"].between(1, 7))]
    df = df.dropna(subset=["birthyr", "gender4", "educ", "race", "urbancity"])

    rng = np.random.default_rng(RNG_SEED)
    if len(df) > N_VOTER_SAMPLE:
        weights = df["commonweight"].fillna(df["commonweight"].median())
        weights = weights.clip(lower=0.01)
        idx = rng.choice(df.index, size=N_VOTER_SAMPLE, replace=False,
                          p=weights / weights.sum())
        df = df.loc[idx]


    df["econ_ideology"] = (df["ideo5"] - 3) / 2.0


    df["partisan_lean"] = (df["pid7"] - 4) / 3.0
    df["partisan_strength"] = df["partisan_lean"].abs()


    educ_component = (df["educ"] - df["educ"].mean()) / df["educ"].std()
    urban_component = (df["urbancity"] - df["urbancity"].mean()) / df["urbancity"].std()
    noise = rng.normal(0, 0.35, size=len(df))
    social_raw = -0.5 * educ_component - 0.3 * urban_component + noise
    df["social_ideology"] = np.clip(social_raw / social_raw.std(), -1, 1)


    base = np.where(df["votereg"] == 1, 0.78, 0.30)
    age = 2024 - df["birthyr"]
    age_boost = np.clip((age - 25) / 100, -0.1, 0.15)
    strength_boost = df["partisan_strength"] * 0.12
    df["turnout_propensity"] = np.clip(base + age_boost + strength_boost, 0.05, 0.97)

    out = df[["econ_ideology", "social_ideology", "partisan_lean",
              "partisan_strength", "turnout_propensity",
              "gender4", "educ", "race", "urbancity"]].reset_index(drop=True)
    out.rename(columns={"gender4": "gender", "urbancity": "urban_rural"}, inplace=True)
    out.to_csv(OUT_DIR / "processed_voters.csv", index=False)
    print(f"  -> saved {len(out)} voter profiles to data/processed_voters.csv")



# processed_parties_uk.csv

CHES_COUNTRY_NAMES = {
    1: "Belgium", 2: "Denmark", 3: "Germany", 4: "Greece", 5: "Spain",
    6: "France", 7: "Ireland", 8: "Italy", 10: "Netherlands",
    11: "United Kingdom", 12: "Portugal", 13: "Austria", 14: "Finland",
    16: "Sweden", 20: "Bulgaria", 21: "Czech Republic", 22: "Estonia",
    23: "Hungary", 24: "Latvia", 25: "Lithuania", 26: "Poland",
    27: "Romania", 28: "Slovakia", 29: "Slovenia", 31: "Croatia",
    37: "Norway", 38: "Switzerland", 40: "Malta",
}


def process_ches():
    print("Processing CHES party-position data ...")
    df = pd.read_csv(CHES_FILE, usecols=[
        "year", "country", "party", "vote", "seat", "lrgen", "galtan", "family"
    ])
    df["country_name"] = df["country"].map(CHES_COUNTRY_NAMES)
    df = df.dropna(subset=["lrgen", "galtan", "vote"])

    # normalise CHES's 0-10 scales onto the same [-1, 1] space used for voters
    df["econ_ideology"] = (df["lrgen"] - 5) / 5.0
    df["social_ideology"] = (df["galtan"] - 5) / 5.0

    df.to_csv(OUT_DIR / "processed_parties_all.csv", index=False)

    uk = df[(df["country"] == 11) & (df["year"] == 2024)].copy()
    uk = uk.sort_values("vote", ascending=False).reset_index(drop=True)
    uk["seat"] = uk["seat"].fillna(0)
    uk.to_csv(OUT_DIR / "processed_parties_uk.csv", index=False)
    print(f"  -> saved {len(df)} party-years to data/processed_parties_all.csv")
    print(f"  -> saved {len(uk)} UK 2024 parties to data/processed_parties_uk.csv")



# uk_ge2024_validation.csv

def process_mps():
    print("Processing UK GE2024 defeated-MP data ...")
    df = pd.read_csv(MPS_FILE, encoding="utf-8-sig")

    def parse_winner(result):

        m = re.match(r"^(.*?)\s+(gain from|hold)", str(result))
        return m.group(1).strip() if m else None

    df["winning_party"] = df["result"].apply(parse_winner)
    df["previous_party"] = df["party_abbreviation"]
    df["seat_changed_hands"] = df["result"].str.contains("gain", case=False, na=False)

    summary = (df.groupby("winning_party")
                 .size()
                 .reset_index(name="seats_gained_from_defeated_mps")
                 .sort_values("seats_gained_from_defeated_mps", ascending=False))

    out_cols = ["ons_id", "constituency_name", "region_name", "country_name",
                "party_abbreviation", "result", "winning_party",
                "previous_party", "seat_changed_hands"]
    df[out_cols].to_csv(OUT_DIR / "uk_ge2024_validation.csv", index=False)
    summary.to_csv(OUT_DIR / "uk_ge2024_validation_summary.csv", index=False)
    print(f"  -> saved {len(df)} defeated-MP records to data/uk_ge2024_validation.csv")


if __name__ == "__main__":
    process_cces()
    process_ches()
    process_mps()
    print("\nAll processed datasets written to:", OUT_DIR.resolve())
