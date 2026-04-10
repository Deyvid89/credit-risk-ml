import numpy as np


def feature_engineering(X):
    X = X.copy()

    # ── LOG ────────────────────────────────────────────────
    X['log_income'] = np.log1p(X['monthly_income'])
    X['log_debt']   = np.log1p(X['debt_ratio'])

    # ── RENDA: features expressivas ────────────────────────
    # Flag: renda zero ou ausente
    X['zero_income'] = (X['monthly_income'] == 0).astype(int)

    # Flag: renda muito baixa (abaixo do 10º percentil ≈ R$ 2.500 no dataset)
    X['low_income'] = (X['monthly_income'] < 2500).astype(int)

    # Dívida absoluta estimada (debt_ratio × renda) — captura endividamento real
    # Para renda zero: debt_ratio perde significado, então usamos 0
    X['estimated_debt'] = X['debt_ratio'] * X['monthly_income']
    X['log_estimated_debt'] = np.log1p(X['estimated_debt'])

    # ── BINÁRIAS ───────────────────────────────────────────
    X['has_late_90']    = (X['late_90_days'] > 0).astype(int)
    X['severe_late_90'] = (X['late_90_days'] >= 2).astype(int)
    X['has_late_30']    = (X['late_30_59_days'] > 0).astype(int)
    X['has_late_60']    = (X['late_60_89_days'] > 0).astype(int)
    X['extreme_debt']      = (X['debt_ratio'] > 0.8).astype(int)
    X['high_credit_util']  = (X['credit_utilization'] > 0.9).astype(int)

    # ── INTERAÇÕES ─────────────────────────────────────────
    X['late_total'] = (
        X['late_30_59_days'] +
        X['late_60_89_days'] +
        X['late_90_days']
    )

    # Parcela de renda comprometida com atrasos (proxy de pressão financeira)
    # late_total já existe aqui, então a feature é calculada corretamente
    X['income_pressure'] = X['debt_ratio'] * (1 + X['late_total'])

    X['risk_score_simple'] = (
        X['has_late_90'] +
        X['severe_late_90'] +
        X['extreme_debt'] +
        X['high_credit_util'] +
        X['zero_income']       # renda zero entra como fator de risco explícito
    )

    # Risco combinado: atraso grave + renda baixa
    X['high_risk_low_income'] = (
        (X['has_late_90'] == 1) & (X['low_income'] == 1)
    ).astype(int)

    # ── CAP ────────────────────────────────────────────────
    X['late_30_59_days'] = X['late_30_59_days'].clip(0, 10)
    X['late_60_89_days'] = X['late_60_89_days'].clip(0, 10)
    X['late_90_days']    = X['late_90_days'].clip(0, 10)

    # ── DROP ───────────────────────────────────────────────
    # Removemos monthly_income e debt_ratio brutos pois foram substituídos
    # pelas versões transformadas e pelas features derivadas acima
    X = X.drop(columns=['monthly_income', 'debt_ratio', 'estimated_debt'])

    return X