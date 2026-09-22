

import numpy as np


def vote_seat_shares(parties):
    total_votes = sum(p.vote_count for p in parties) or 1
    total_seats = sum(p.seats for p in parties) or 1
    for p in parties:
        p.vote_share = p.vote_count / total_votes
        p.seat_share = p.seats / total_seats
    return {p.name: (p.vote_share, p.seat_share) for p in parties}


def gallagher_index(parties):
    total_votes = sum(p.vote_count for p in parties)
    total_seats = sum(p.seats for p in parties)
    if total_votes == 0 or total_seats == 0:
        return 0.0
    sq_diffs = []
    for p in parties:
        v_pct = 100 * p.vote_count / total_votes
        s_pct = 100 * p.seats / total_seats
        sq_diffs.append((v_pct - s_pct) ** 2)
    return float(np.sqrt(0.5 * sum(sq_diffs)))


def effective_number_of_parties(shares):

    shares = [s for s in shares if s > 0]
    if not shares:
        return 0.0
    return float(1.0 / sum(s ** 2 for s in shares))


def turnout_rate(voters):
    if not voters:
        return 0.0
    return float(sum(1 for v in voters if v.voted) / len(voters))


def polarisation_index(voters):

    if not voters:
        return 0.0
    pos = np.array([v.position for v in voters])
    centroid = pos.mean(axis=0)
    dists = np.linalg.norm(pos - centroid, axis=1)
    return float(dists.std())


def bimodality_coefficient(voters, axis=0):

    if len(voters) < 4:
        return 0.0
    from scipy.stats import skew, kurtosis
    x = np.array([v.position[axis] for v in voters])
    n = len(x)
    g = skew(x)
    k = kurtosis(x, fisher=False)
    bc = (g ** 2 + 1) / (k + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))) if n > 3 else 0.0
    return float(bc)


def ideological_diversity(parties):

    active = [p for p in parties if p.seats > 0]
    if len(active) < 2:
        return 0.0
    total = 0.0
    weight_sum = 0.0
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            d = np.linalg.norm(active[i].position - active[j].position)
            w = active[i].seat_share * active[j].seat_share
            total += d * w
            weight_sum += w
    return float(total / weight_sum) if weight_sum > 0 else 0.0


def summarise_election(voters, parties):
    vote_seat_shares(parties)
    return {
        "turnout": turnout_rate(voters),
        "gallagher_index": gallagher_index(parties),
        "enp_votes": effective_number_of_parties([p.vote_share for p in parties]),
        "enp_seats": effective_number_of_parties([p.seat_share for p in parties]),
        "polarisation": polarisation_index(voters),
        "ideological_diversity": ideological_diversity(parties),
    }
