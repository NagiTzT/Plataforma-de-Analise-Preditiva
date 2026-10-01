"""Features futebolísticas compartilhadas por app e robô.

Todas as funções recebem apenas informações disponíveis antes da partida.
"""

from __future__ import annotations

from typing import Iterable, Sequence
import math


RECENCY_WEIGHTS = (1.00, .90, .82, .74, .68, .55, .48, .42, .37, .33)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def enhanced_rolling_features(
    jogos: Iterable[Sequence[float]], prefix: str, draw_prior: float = .27,
    ppg_prior: float = 1.35,
) -> dict[str, float]:
    """Resume até 10 jogos, do mais recente para o mais antigo.

    Cada item contém ``(pontos, gols_pro, gols_contra[, ppg_adversario])``.
    A forma ponderada é regredida para a média quando há menos de cinco jogos.
    """
    rows = [tuple(item) for item in jogos][:10]
    if not rows:
        return {
            f"{prefix}_ppg": ppg_prior, f"{prefix}_weighted_ppg": ppg_prior,
            f"{prefix}_shrunk_ppg": ppg_prior,
            f"{prefix}_opponent_ppg": ppg_prior,
            f"{prefix}_strength_adjusted_ppg": ppg_prior,
            f"{prefix}_gf": 1.30, f"{prefix}_ga": 1.30,
            f"{prefix}_saldo": 0.0, f"{prefix}_win_rate": 0.0,
            f"{prefix}_draw_rate": float(draw_prior),
            f"{prefix}_games": 0.0, f"{prefix}_available": 0.0,
            f"{prefix}_sample_reliability": 0.0,
        }
    n = len(rows)
    weights = RECENCY_WEIGHTS[:n]
    weight_sum = sum(weights)
    points = [float(row[0]) for row in rows]
    gf = [float(row[1]) for row in rows]
    ga = [float(row[2]) for row in rows]
    opponents = [float(row[3]) if len(row) > 3 else ppg_prior for row in rows]
    raw_ppg = sum(points) / n
    weighted_ppg = sum(value * weight for value, weight in zip(points, weights)) / weight_sum
    reliability = _clamp(n / 5.0, 0.0, 1.0)
    shrunk_ppg = reliability * weighted_ppg + (1.0 - reliability) * ppg_prior
    opponent_ppg = sum(value * weight for value, weight in zip(opponents, weights)) / weight_sum
    adjusted_ppg = _clamp(shrunk_ppg + .30 * (opponent_ppg - ppg_prior), 0.0, 3.0)
    return {
        f"{prefix}_ppg": raw_ppg,
        f"{prefix}_weighted_ppg": weighted_ppg,
        f"{prefix}_shrunk_ppg": shrunk_ppg,
        f"{prefix}_opponent_ppg": opponent_ppg,
        f"{prefix}_strength_adjusted_ppg": adjusted_ppg,
        f"{prefix}_gf": sum(gf) / n,
        f"{prefix}_ga": sum(ga) / n,
        f"{prefix}_saldo": (sum(gf) - sum(ga)) / n,
        f"{prefix}_win_rate": sum(value == 3 for value in points) / n,
        f"{prefix}_draw_rate": sum(value == 1 for value in points) / n,
        f"{prefix}_games": float(n), f"{prefix}_available": 1.0,
        f"{prefix}_sample_reliability": reliability,
    }


def add_venue_comparison(features: dict[str, float]) -> dict[str, float]:
    """Mede se casa/fora altera o time, sem conceder bônus fixo ao mandante."""
    def value(name: str, default: float = 1.35) -> float:
        try:
            return float(features.get(name, default) or 0.0)
        except (TypeError, ValueError):
            return default

    home_general = value("form_home_10_strength_adjusted_ppg")
    away_general = value("form_away_10_strength_adjusted_ppg")
    home_venue = value("form_home_casa_10_strength_adjusted_ppg")
    away_venue = value("form_away_fora_10_strength_adjusted_ppg")
    home_rel = value("form_home_casa_10_sample_reliability", 0.0)
    away_rel = value("form_away_fora_10_sample_reliability", 0.0)
    features.update({
        "context_mando_home_delta": (home_venue - home_general) * home_rel,
        "context_mando_away_delta": (away_venue - away_general) * away_rel,
        "context_mando_ajustado_gap": (
            home_venue * home_rel + home_general * (1.0 - home_rel)
            - away_venue * away_rel - away_general * (1.0 - away_rel)
        ),
        "context_mando_amostra_min": min(home_rel, away_rel),
    })
    return features


def add_measured_sofa_duel(features: dict[str, float]) -> dict[str, float]:
    """An xG duel requires measured samples for both attack/defense pairs.

    Missing measurements never imply zero expected goals or perfect parity.
    New snapshots carry counts; old snapshots without counts remain unknown.
    """
    def value(key, default=0.0):
        try:
            result = float(features.get(key, default))
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default
    count = min(value('sofa_roll_home_xg_games'), value('sofa_roll_away_xg_games'))
    available = count > 0
    expected_home = (value('sofa_roll_home_xg_for_avg') + value('sofa_roll_away_xg_against_avg')) / 2
    expected_away = (value('sofa_roll_away_xg_for_avg') + value('sofa_roll_home_xg_against_avg')) / 2
    total = expected_home + expected_away
    parity = math.exp(-abs(expected_home - expected_away) / .75) if available else 0.0
    draw_history = (value('sofa_roll_home_draw_rate', .27) + value('sofa_roll_away_draw_rate', .27)) / 2
    low_total = max(0.0, min(1.0, (3.0-total)/1.8)) if available else 0.0
    features.update({
        'sofa_roll_duel_available': float(available),
        'sofa_roll_duel_xg_sample_min': count,
        'sofa_roll_duel_xg_reliability': min(1.0, count / 5),
        'sofa_roll_duel_expected_home': expected_home if available else 0.0,
        'sofa_roll_duel_expected_away': expected_away if available else 0.0,
        'sofa_roll_duel_expected_total': total if available else 0.0,
        'sofa_roll_duel_attack_gap': expected_home - expected_away if available else 0.0,
        'sofa_roll_duel_xg_parity': parity,
        'sofa_roll_duel_draw_signal': (.45*draw_history+.30*parity+.25*low_total) if available else 0.0,
    })
    for metric, name in (('shots', 'total_shots'),
                         ('shots_on_target', 'shots'),
                         ('big_chances', 'big_chances'),
                         ('box_shots', 'box_shots')):
        measured = min(value(f'sofa_roll_{side}_{metric}_{direction}_games')
                       for side in ('home', 'away') for direction in ('for', 'against')) > 0
        features[f'sofa_roll_duel_{name}_available'] = float(measured)
        features[f'sofa_roll_duel_{name}_gap'] = (
            value(f'sofa_roll_home_{metric}_for_avg') + value(f'sofa_roll_away_{metric}_against_avg')
            - value(f'sofa_roll_away_{metric}_for_avg') - value(f'sofa_roll_home_{metric}_against_avg')
        ) / 2 if measured else 0.0
    possession_games = min(value(f'sofa_roll_{side}_possession_games')
                           for side in ('home', 'away'))
    features['sofa_roll_duel_possession_available'] = float(possession_games > 0)
    features['sofa_roll_duel_possession_gap'] = (
        value('sofa_roll_home_possession_avg')
        - value('sofa_roll_away_possession_avg')
    ) if possession_games > 0 else 0.0
    # New coverage-aware windows. Five games capture current momentum; ten
    # reduce variance. Both remain separate so a learner can use disagreement
    # as a trend rather than replacing one with the other.
    for window in (5, 10):
        suffix = str(window)
        home_goal_count = value(f'sofa_roll_home_goals_for_games_{suffix}')
        away_goal_count = value(f'sofa_roll_away_goals_for_games_{suffix}')
        home_ga_count = value(f'sofa_roll_home_goals_against_games_{suffix}')
        away_ga_count = value(f'sofa_roll_away_goals_against_games_{suffix}')
        goal_sample = min(home_goal_count, away_goal_count, home_ga_count, away_ga_count)
        goal_available = goal_sample >= 3
        goal_expected_home = (
            value(f'sofa_roll_home_goals_for_opponent_adjusted_{suffix}')
            + value(f'sofa_roll_away_goals_against_opponent_adjusted_{suffix}')
        ) / 2
        goal_expected_away = (
            value(f'sofa_roll_away_goals_for_opponent_adjusted_{suffix}')
            + value(f'sofa_roll_home_goals_against_opponent_adjusted_{suffix}')
        ) / 2
        goal_total = goal_expected_home + goal_expected_away
        goal_parity = math.exp(-abs(goal_expected_home-goal_expected_away)) if goal_available else 0.0
        poisson_draw = sum(
            math.exp(-goal_total) * (goal_expected_home*goal_expected_away)**k
            / math.factorial(k)**2 for k in range(12)
        ) if goal_available else 0.0
        draw_history = (
            value(f'sofa_roll_home_draw_rate_weighted_{suffix}')
            + value(f'sofa_roll_away_draw_rate_weighted_{suffix}')
        ) / 2
        low_total_history = (
            value(f'sofa_roll_home_low_total_rate_{suffix}')
            + value(f'sofa_roll_away_low_total_rate_{suffix}')
        ) / 2
        close_history = (
            value(f'sofa_roll_home_close_game_rate_{suffix}')
            + value(f'sofa_roll_away_close_game_rate_{suffix}')
        ) / 2
        features.update({
            f'sofa_roll_duel_goal_window_{suffix}_available': float(goal_available),
            f'sofa_roll_duel_goal_window_{suffix}_sample_min': goal_sample,
            f'sofa_roll_duel_goal_window_{suffix}_expected_home': goal_expected_home if goal_available else 0.0,
            f'sofa_roll_duel_goal_window_{suffix}_expected_away': goal_expected_away if goal_available else 0.0,
            f'sofa_roll_duel_goal_window_{suffix}_attack_gap': (
                goal_expected_home-goal_expected_away if goal_available else 0.0
            ),
            f'sofa_roll_duel_goal_window_{suffix}_expected_total': goal_total if goal_available else 0.0,
            f'sofa_roll_duel_goal_window_{suffix}_parity': goal_parity,
            f'sofa_roll_duel_goal_window_{suffix}_draw_poisson': poisson_draw,
            f'sofa_roll_duel_goal_window_{suffix}_draw_history': draw_history if goal_available else 0.0,
            f'sofa_roll_duel_goal_window_{suffix}_low_total_history': low_total_history if goal_available else 0.0,
            f'sofa_roll_duel_goal_window_{suffix}_close_history': close_history if goal_available else 0.0,
        })
        home_xg_count = value(f'sofa_roll_home_xg_for_games_{suffix}')
        away_xg_count = value(f'sofa_roll_away_xg_for_games_{suffix}')
        home_xga_count = value(f'sofa_roll_home_xg_against_games_{suffix}')
        away_xga_count = value(f'sofa_roll_away_xg_against_games_{suffix}')
        sample = min(home_xg_count, away_xg_count, home_xga_count, away_xga_count)
        window_available = sample >= 3
        expected_home = (
            value(f'sofa_roll_home_xg_for_opponent_adjusted_{suffix}')
            + value(f'sofa_roll_away_xg_against_opponent_adjusted_{suffix}')
        ) / 2
        expected_away = (
            value(f'sofa_roll_away_xg_for_opponent_adjusted_{suffix}')
            + value(f'sofa_roll_home_xg_against_opponent_adjusted_{suffix}')
        ) / 2
        features.update({
            f'sofa_roll_duel_window_{suffix}_available': float(window_available),
            f'sofa_roll_duel_window_{suffix}_sample_min': sample,
            f'sofa_roll_duel_window_{suffix}_expected_home': expected_home if window_available else 0.0,
            f'sofa_roll_duel_window_{suffix}_expected_away': expected_away if window_available else 0.0,
            f'sofa_roll_duel_window_{suffix}_attack_gap': (
                expected_home - expected_away if window_available else 0.0
            ),
            f'sofa_roll_duel_window_{suffix}_expected_total': (
                expected_home + expected_away if window_available else 0.0
            ),
        })
        for style in ('attack_intensity', 'chance_quality', 'box_shot_share',
                      'territorial_control', 'pressure_proxy', 'directness'):
            both = min(value(f'sofa_roll_home_style_available_{suffix}'),
                       value(f'sofa_roll_away_style_available_{suffix}')) > 0
            features[f'sofa_roll_duel_{style}_available_{suffix}'] = float(both)
            features[f'sofa_roll_duel_{style}_gap_{suffix}'] = (
                value(f'sofa_roll_home_{style}_{suffix}')
                - value(f'sofa_roll_away_{style}_{suffix}') if both else 0.0
            )
    if (features.get('sofa_roll_duel_window_5_available')
            and features.get('sofa_roll_duel_window_10_available')):
        features['sofa_roll_duel_attack_momentum'] = (
            features['sofa_roll_duel_window_5_attack_gap']
            - features['sofa_roll_duel_window_10_attack_gap']
        )
    else:
        features['sofa_roll_duel_attack_momentum'] = 0.0
    return features


__all__ = ["RECENCY_WEIGHTS", "enhanced_rolling_features", "add_venue_comparison", "add_measured_sofa_duel"]
