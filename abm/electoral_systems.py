

import numpy as np
from collections import defaultdict


def _party_by_name(parties):
    return {p.name: p for p in parties}


def assign_districts(voters, n_districts, rng):

    positions = np.array([v.position for v in voters])

    score = positions[:, 0] + 0.5 * positions[:, 1] + rng.normal(0, 0.6, len(voters))
    order = np.argsort(score)
    chunks = np.array_split(order, n_districts)
    district_of = {}
    for d, chunk in enumerate(chunks):
        for idx in chunk:
            district_of[voters[idx].unique_id] = d
    for v in voters:
        v.district = district_of[v.unique_id]
    return district_of


def dhondt(votes, total_seats, threshold=0.0):

    total_votes = sum(votes.values())
    eligible = {p: v for p, v in votes.items()
                if total_votes > 0 and v / total_votes >= threshold}
    seats = {p: 0 for p in votes}
    if not eligible or total_seats == 0:
        return seats
    for _ in range(total_seats):
        quotients = {p: v / (seats[p] + 1) for p, v in eligible.items()}
        winner = max(quotients, key=quotients.get)
        seats[winner] += 1
    return seats


def sainte_lague(votes, total_seats, threshold=0.0):

    total_votes = sum(votes.values())
    eligible = {p: v for p, v in votes.items()
                if total_votes > 0 and v / total_votes >= threshold}
    seats = {p: 0 for p in votes}
    if not eligible or total_seats == 0:
        return seats
    for _ in range(total_seats):
        quotients = {p: v / (2 * seats[p] + 1) for p, v in eligible.items()}
        winner = max(quotients, key=quotients.get)
        seats[winner] += 1
    return seats


def run_fptp(voters, parties, n_districts, rng, **kwargs):
    voted = [v for v in voters if v.voted and v.party_choice is not None]
    if not voted:
        return {p.name: 0 for p in parties}
    assign_districts(voted, n_districts, rng)
    id_to_name = {p.unique_id: p.name for p in parties}
    seats = defaultdict(int)
    for d in range(n_districts):
        in_d = [v for v in voted if v.district == d]
        if not in_d:
            continue
        tally = defaultdict(int)
        for v in in_d:
            tally[id_to_name[v.party_choice]] += 1
        winner = max(tally, key=tally.get)
        seats[winner] += 1
    return {p.name: seats.get(p.name, 0) for p in parties}


def run_pr(voters, parties, total_seats, rng, threshold=0.05, **kwargs):
    voted = [v for v in voters if v.voted and v.party_choice is not None]
    id_to_name = {p.unique_id: p.name for p in parties}
    tally = defaultdict(int)
    for v in voted:
        tally[id_to_name[v.party_choice]] += 1
    for p in parties:
        tally.setdefault(p.name, 0)
    seats = dhondt(dict(tally), total_seats, threshold=threshold)
    return seats


def run_mmp(voters, parties, n_districts, total_seats, rng, threshold=0.05, **kwargs):

    voted = [v for v in voters if v.voted and v.party_choice is not None]
    id_to_name = {p.unique_id: p.name for p in parties}

    constituency_seats_n = max(1, total_seats // 2)
    list_seats_n = total_seats - constituency_seats_n

    const_seats = run_fptp(voters, parties, constituency_seats_n, rng)


    tally = defaultdict(int)
    for v in voted:
        tally[id_to_name[v.party_choice]] += 1
    for p in parties:
        tally.setdefault(p.name, 0)


    target_total = sainte_lague(dict(tally), total_seats, threshold=threshold)

    final_seats = {}
    remaining_list = list_seats_n

    for name in tally:
        final_seats[name] = max(const_seats.get(name, 0), 0)


    gaps = {name: max(0, target_total.get(name, 0) - final_seats.get(name, 0))
            for name in tally}
    total_gap = sum(gaps.values())
    if total_gap > 0 and remaining_list > 0:
        for name, gap in sorted(gaps.items(), key=lambda x: -x[1]):
            if remaining_list <= 0:
                break
            grant = min(gap, remaining_list) if total_gap <= remaining_list else \
                round(remaining_list * gap / total_gap)
            grant = min(grant, remaining_list)
            final_seats[name] += grant
            remaining_list -= grant

        if remaining_list > 0:
            biggest = max(tally, key=tally.get)
            final_seats[biggest] += remaining_list

    return {p.name: final_seats.get(p.name, 0) for p in parties}


SYSTEMS = {
    "FPTP": run_fptp,
    "PR": run_pr,
    "MMP": run_mmp,
}
