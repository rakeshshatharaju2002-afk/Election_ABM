

import numpy as np
from mesa import Agent


class VoterAgent(Agent):
    def __init__(self, model, econ_ideology, social_ideology,
                 partisan_lean, partisan_strength, turnout_propensity,
                 confidence_eps=0.45, susceptibility=0.15):
        super().__init__(model)
        self.econ_ideology = float(econ_ideology)
        self.social_ideology = float(social_ideology)
        self.partisan_lean = float(partisan_lean)
        self.partisan_strength = float(partisan_strength)
        self.base_turnout_propensity = float(turnout_propensity)
        self.confidence_eps = confidence_eps      # bounded-confidence threshold
        self.susceptibility = susceptibility       # how much it moves per cycle
        self.district = None
        self.party_choice = None
        self.voted = False

    @property
    def position(self):
        return np.array([self.econ_ideology, self.social_ideology])


    def update_opinion(self, neighbor_positions):

        if len(neighbor_positions) == 0:
            return
        my_pos = self.position
        close = [p for p in neighbor_positions
                 if np.linalg.norm(p - my_pos) < self.confidence_eps]
        if not close:
            return
        target = np.mean(close, axis=0)
        effective_susceptibility = self.susceptibility * (1 - 0.6 * self.partisan_strength)
        new_pos = my_pos + effective_susceptibility * (target - my_pos)

        new_pos = new_pos + self.model.rng.normal(0, 0.01, size=2)
        new_pos = np.clip(new_pos, -1, 1)
        self.econ_ideology, self.social_ideology = new_pos


    def choose_party(self, parties):
        my_pos = self.position
        scores = []
        for p in parties:
            dist = np.linalg.norm(my_pos - p.position)

            lean_bonus = 0.15 * abs(self.partisan_lean - p.partisan_axis_value) * -1
            score = -dist + p.valence + lean_bonus
            scores.append(score)
        best = int(np.argmax(scores))
        self.party_choice = parties[best].unique_id
        return parties[best]


    def decide_turnout(self, satisfaction_bonus=0.0):
        p = np.clip(self.base_turnout_propensity + satisfaction_bonus, 0.02, 0.99)
        self.voted = self.model.rng.random() < p
        return self.voted


class PartyAgent(Agent):
    def __init__(self, model, name, econ_ideology, social_ideology,
                 adapt_rate=0.08, valence=0.0):
        super().__init__(model)
        self.name = name
        self.econ_ideology = float(econ_ideology)
        self.social_ideology = float(social_ideology)
        self.adapt_rate = adapt_rate
        self.valence = valence
        self.vote_count = 0
        self.vote_share = 0.0
        self.seats = 0
        self.seat_share = 0.0
        self.history = []

    @property
    def position(self):
        return np.array([self.econ_ideology, self.social_ideology])

    @property
    def partisan_axis_value(self):

        return self.econ_ideology

    def adapt_position(self, supporter_positions):

        if len(supporter_positions) == 0:
            self.history.append(tuple(self.position))
            return
        centroid = np.mean(supporter_positions, axis=0)
        new_pos = self.position + self.adapt_rate * (centroid - self.position)
        new_pos = np.clip(new_pos, -1, 1)
        self.econ_ideology, self.social_ideology = new_pos
        self.history.append(tuple(self.position))
