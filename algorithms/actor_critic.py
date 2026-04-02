"""One-step Actor-Critic and Actor-Critic with eligibility traces AC(lambda)."""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent


class ActorCriticAgent(Agent):
    raise NotImplementedError


class ActorCriticLambdaAgent(Agent):
    raise NotImplementedError
