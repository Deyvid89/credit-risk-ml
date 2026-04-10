import numpy as np


def logistic(x: float, k: float = 1) -> float:
    """Função logística com inclinação k."""
    return 1 / (1 + np.exp(-k * x))


def apply_business_rules(prob_model: float, inputs: dict) -> tuple[float, list]:
    """
    Ajusta a probabilidade do modelo com regras de negócio baseadas em
    três fatores independentes: atrasos, endividamento e renda.

    Cada fator gera um índice contínuo [0, 1] e um piso de probabilidade
    próprio — sem fronteiras duras ou saltos bruscos. O maior piso prevalece.

    Retorna (probabilidade final, lista de alertas para exibição).
    """
    # ── ALERTAS (apenas para exibição na interface) ─────────
    # Cada regra é um dict com: nível, mensagem e condição (lambda sobre inputs)
    # Regras mutuamente exclusivas usam "exclusive_group" — só a primeira
    # condição verdadeira do grupo dispara.
    ALERT_RULES = [
        {
            "level": "err",
            "msg":   "⚠️ Atrasos de 90 dias detectados",
            "cond":  lambda i: i["late_90"] > 0,
        },
        {
            "level": "warn",
            "msg":   "🕐 Atrasos de 60 dias no histórico",
            "cond":  lambda i: i["late_60"] > 0,
        },
        {
            "level": "warn",
            "msg":   "🕐 Atrasos de 30 dias no histórico",
            "cond":  lambda i: i["late_30"] > 0,
        },
        {
            "level": "err",
            "msg":   "📉 Endividamento crítico detectado",
            "cond":  lambda i: i["debt_ratio"] > 0.8 and i["credit_utilization"] > 0.8,
        },
        {
            "level": "warn",
            "msg":   "💳 Alto uso de crédito",
            "cond":  lambda i: i["credit_utilization"] > 0.7,
        },
        {
            "level": "warn",
            "msg":   "📊 Endividamento elevado",
            "cond":  lambda i: i["debt_ratio"] > 0.5,
        },
        # Grupo exclusivo de renda — só o alerta mais grave dispara
        {
            "level": "err",
            "msg":   "🚫 Renda zero — capacidade de pagamento não identificada",
            "cond":  lambda i: i["monthly_income"] == 0,
            "exclusive_group": "income",
        },
        {
            "level": "warn",
            "msg":   "💸 Renda muito baixa — risco elevado de inadimplência",
            "cond":  lambda i: i["monthly_income"] < 1500,
            "exclusive_group": "income",
        },
        {
            "level": "warn",
            "msg":   "💸 Renda baixa em relação ao perfil de endividamento",
            "cond":  lambda i: i["monthly_income"] < 3000,
            "exclusive_group": "income",
        },
    ]

    triggered = []
    fired_groups: set = set()

    for rule in ALERT_RULES:
        group = rule.get("exclusive_group")
        if group and group in fired_groups:
            continue
        if rule["cond"](inputs):
            triggered.append((rule["level"], rule["msg"]))
            if group:
                fired_groups.add(group)

    # ══════════════════════════════════════════════════════════
    # ÍNDICES CONTÍNUOS [0, 1] POR FATOR
    # ══════════════════════════════════════════════════════════

    # 1. ATRASOS — ponderado por gravidade (90d×3, 60d×2.5, 30d×1)
    late_raw = inputs["late_90"] * 3 + inputs["late_60"] * 2.5 + inputs["late_30"] * 1
    late_index = np.clip(late_raw / 60.0, 0, 1)
    late_component = np.sqrt(late_index)

    # 2. ENDIVIDAMENTO — logística sobre debt_ratio + credit_utilization
    debt_util = inputs["debt_ratio"] + inputs["credit_utilization"]
    debt_component = logistic(debt_util - 1.3, k=6)

    # 3. RENDA — log-normalizada invertida (renda baixa → componente alto)
    max_income_ref = 10000.0
    income_norm = 1.0 - np.clip(
        np.log1p(inputs["monthly_income"]) / np.log1p(max_income_ref), 0, 1
    )
    income_component = logistic(income_norm - 0.3, k=8)

    # ══════════════════════════════════════════════════════════
    # PISOS INDEPENDENTES — o maior prevalece
    # ══════════════════════════════════════════════════════════

    # Atrasos: piso máximo 65%
    # 1 atraso 60d → ~17% | 1 atraso 90d → ~46% | máximo → 65%
    late_floor = 0.65 * np.sqrt(late_index)

    # Endividamento: piso máximo 25% (nunca Excelente com dívida+uso críticos)
    # ambos 0.90 → ~24% | ambos 0.65 → ~12% | ambos 0.50 → ~3%
    debt_floor = 0.25 * logistic(debt_util - 1.3, k=6)

    # Renda: piso máximo 55% (renda zero = sempre Alto Risco)
    # renda 0 → ~55% | renda 1500 → ~32% | renda 3000 → ~10%
    income_floor = 0.55 * logistic(income_norm - 0.4, k=10)

    floor_prob = float(np.clip(max(late_floor, debt_floor, income_floor), 0, 0.65))

    # ══════════════════════════════════════════════════════════
    # AJUSTE ADAPTATIVO DO MODELO
    # ALPHA decresce conforme o modelo já está confiante,
    # e a amplificação é limitada a 1.4× a prob original.
    # ══════════════════════════════════════════════════════════
    combined_index = (
        late_component   * 0.55 +
        debt_component   * 0.30 +
        income_component * 0.15
    )
    logit_model = np.log(prob_model / (1 - prob_model + 1e-6))
    alpha = 0.30 * (1 - 0.5 * prob_model)
    adjusted_prob = 1 / (1 + np.exp(-(logit_model + alpha * combined_index * 2)))
    adjusted_prob = float(np.clip(adjusted_prob, 0, min(prob_model * 1.4, 1.0)))

    final_prob = max(adjusted_prob, floor_prob)
    return final_prob, triggered