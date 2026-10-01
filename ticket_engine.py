"""Regras puras e testáveis para montar bilhetes de quatro jogos."""

import math


def _score_individual(jogo):
    """Ordena sem excluir: confiança + cobertura - conflito/empate ignorado.

    Agrupar as seleções mais consistentes entre si concentra as partidas mais
    frágeis nos últimos bilhetes, em vez de contaminar todos os grupos 4/4.
    """
    confidence = max(0.0, min(1.0, float(jogo.get("Confiança", 0)) / 100.0))
    quality = max(0.0, min(1.0, float(jogo.get("Qualidade_Contexto", 0) or 0)))
    conflict = max(0.0, min(1.0, float(jogo.get("Conflito_Contexto", 0) or 0)))
    disagreement = max(0.0, min(1.0, float(jogo.get("Divergencia_Fontes", 0) or 0)))
    draw_risk = max(0.0, min(1.0, float(jogo.get("Risco_Empate", 0) or 0)))
    sample = max(0.0, min(1.0, float(jogo.get("Confiabilidade_Amostra", quality) or 0)))
    margin = max(0.0, min(1.0, float(jogo.get("Margem_Probabilidade", 0.08) or 0)))
    entropy = max(0.0, min(1.0, float(jogo.get("Entropia_Normalizada", 0.85) or 0)))
    detailed = max(0.0, min(1.0, float(jogo.get("Contexto_Detalhado_Ambos", 0.5) or 0)))
    live_recent = max(0.0, min(1.0, float(jogo.get("Forma_Viva_Ambos", 0) or 0)))
    data_coverage = max(detailed, live_recent)
    volatility = max(0.0, min(1.0, float(jogo.get("Volatilidade_Competicao", 0.08) or 0)))
    pick = str(jogo.get("Pick", "")).upper()
    score = confidence
    score *= 0.82 + 0.18 * quality
    score *= 0.84 + 0.16 * sample
    score *= 0.78 + 0.22 * (1.0 - entropy)
    score *= 0.82 + 0.18 * min(1.0, margin / 0.15)
    score *= 0.94 + 0.06 * data_coverage
    score *= 1.0 - 0.16 * volatility
    score *= 1.0 - 0.30 * conflict
    score *= 1.0 - 0.18 * disagreement
    if pick == "EMPATE":
        score *= 0.85 + 0.50 * draw_risk
    else:
        score *= 1.0 - 0.35 * max(0.0, draw_risk - 0.24)
    return score


def selecionar_grupos_bilhetes(jogos, min_confianca=50, max_bilhetes=0,
                               permitir_empate=False, ligas_unicas=True,
                               relaxar_ligas=False):
    """Seleciona os melhores grupos 4/4 sem reutilizar jogos.

    A função não usa resultados reais nem faz I/O. Um grupo incompleto é
    descartado; não existe fallback que relaxe as regras de qualidade.
    """
    candidatos = []
    for jogo in jogos:
        try:
            confianca = float(jogo.get("Confiança", 0))
        except (TypeError, ValueError):
            continue
        pick = str(jogo.get("Pick", "")).upper()
        if confianca < min_confianca:
            continue
        if not permitir_empate and (pick == "EMPATE" or "EMPATE" in str(jogo.get("Vencedor Escolhido", "")).upper()):
            continue
        candidatos.append(jogo)

    disponiveis = sorted(
        candidatos,
        key=lambda j: (_score_individual(j), float(j.get("Confiança", 0))),
        reverse=True,
    )
    grupos = []
    limite = None if max_bilhetes is None or int(max_bilhetes) <= 0 else int(max_bilhetes)
    # Quando a diversidade de ligas é uma preferência (e não um bloqueio),
    # reserva primeiro os melhores múltiplos de quatro. Assim, se sobrarem
    # 1–3 jogos, os descartados serão sempre os de menor confiança.
    if relaxar_ligas:
        total_grupos = len(disponiveis) // 4
        if limite is not None:
            total_grupos = min(total_grupos, limite)
        disponiveis = disponiveis[:total_grupos * 4]
    while len(disponiveis) >= 4 and (limite is None or len(grupos) < limite):
        grupo, usadas = [], set()
        for jogo in disponiveis:
            liga = str(jogo.get("Liga_Exata", jogo.get("Liga", "Desconhecida")))
            if ligas_unicas and liga in usadas:
                continue
            grupo.append(jogo)
            usadas.add(liga)
            if len(grupo) == 4:
                break
        # Diversidade de ligas reduz correlação, mas não deve impedir um grupo
        # completo quando o operador explicitamente permite a flexibilização.
        if len(grupo) < 4 and ligas_unicas and relaxar_ligas:
            ids_grupo = {str(j.get("ID", j.get("ID_Jogo", ""))) for j in grupo}
            for jogo in disponiveis:
                jogo_id = str(jogo.get("ID", jogo.get("ID_Jogo", "")))
                if jogo_id in ids_grupo:
                    continue
                grupo.append(jogo)
                ids_grupo.add(jogo_id)
                if len(grupo) == 4:
                    break
        if len(grupo) != 4:
            break
        ids = {str(j.get("ID", j.get("ID_Jogo", ""))) for j in grupo}
        confiancas = [max(0.0, min(1.0, float(j.get("Confiança", 0)) / 100.0)) for j in grupo]
        minimum = min(confiancas)
        categoria = (
            "ELITE" if minimum >= 0.50 else
            ("PADRÃO" if minimum >= 0.40 else "ALTO RISCO")
        )
        grupos.append({
            "Jogos": grupo,
            "Probabilidade Conjunta": math.prod(confiancas),
            "Acertos Esperados": sum(confiancas),
            "Score de Consistência": math.prod(max(1e-6, _score_individual(j)) for j in grupo),
            "Categoria": categoria,
        })
        disponiveis = [j for j in disponiveis
                       if str(j.get("ID", j.get("ID_Jogo", ""))) not in ids]
    return grupos

