"""
挑战系统 - Challenges
Kaggle 风格的因果推断竞赛挑战
"""

from .challenge_base import Challenge, ChallengeResult
from .challenge_1_ate_estimation import ATEEstimationChallenge
from .challenge_2_cate_prediction import CATEPredictionChallenge
from .challenge_3_uplift_ranking import UpliftRankingChallenge
from .leaderboard import Leaderboard
from .ui import render

__all__ = [
    'Challenge',
    'ChallengeResult',
    'ATEEstimationChallenge',
    'CATEPredictionChallenge',
    'UpliftRankingChallenge',
    'Leaderboard',
    'render'
]
