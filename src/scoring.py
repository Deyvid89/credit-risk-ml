import numpy as np
import pandas as pd


def validate_inputs(inputs: dict) -> list[str]:
    """Valida os inputs do formulário e retorna lista de erros."""
    errs = []
    if inputs["monthly_income"] < 0:
        errs.append("Renda mensal deve ser ≥ 0.")
    if not 0 <= inputs["credit_utilization"] <= 1:
        errs.append("Uso de crédito deve estar entre 0 e 1.")
    if not 0 <= inputs["debt_ratio"] <= 1:
        errs.append("Taxa de endividamento deve estar entre 0 e 1.")
    return errs


def build_features(inputs: dict) -> pd.DataFrame:
    """Constrói o DataFrame de features brutas para o modelo."""
    df = pd.DataFrame([{
        "credit_utilization": inputs["credit_utilization"],
        "age":                inputs["age"],
        "late_30_59_days":    inputs["late_30"],
        "debt_ratio":         inputs["debt_ratio"],
        "monthly_income":     inputs["monthly_income"],
        "open_credit_lines":  inputs["open_credit_lines"],
        "late_90_days":       inputs["late_90"],
        "real_estate_loans":  inputs["real_estate_loans"],
        "late_60_89_days":    inputs["late_60"],
        "dependents":         inputs["dependents"],
    }])
    return df[[
        "credit_utilization", "age", "late_30_59_days", "debt_ratio",
        "monthly_income", "open_credit_lines", "late_90_days",
        "real_estate_loans", "late_60_89_days", "dependents",
    ]]


def calculate_score(prob: float) -> int:
    """Converte probabilidade de inadimplência em score (0–1000)."""
    return int(1000 / (1 + np.exp(6 * (prob - 0.4))))


def score_label(score: int) -> tuple[str, str, str]:
    """Retorna (label, cor, classe CSS) para o score."""
    if score >= 760:
        return "Excelente", "green", "pill-green"
    elif score >= 600:
        return "Bom", "blue", "pill-blue"
    elif score >= 430:
        return "Moderado", "orange", "pill-orange"
    return "Alto Risco", "red", "pill-red"
