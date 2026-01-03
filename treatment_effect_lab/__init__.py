"""
TreatmentEffectLab - 处理效应估计实验室

包含:
- propensity_score: 倾向得分方法 (PSM)
- ipw: 逆概率加权 (IPW/AIPW)
- doubly_robust: 双重稳健估计
"""

from . import propensity_score
from . import ipw
from . import doubly_robust
