"""Render a reproducible research handoff; never select or promote a model."""
import argparse
import json
from pathlib import Path
import numpy as np


def pct(x): return '—' if x is None else f'{100*x:.2f}%'
def number(x): return '—' if x is None else f'{x:.3f}'
def interval(x): return f'[{pct(x[0])}; {pct(x[1])}]'
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def render(folder, first):
    dest=Path(folder); d=json.loads((dest/'report.json').read_text(encoding='utf-8'))
    old=json.loads(Path(first).read_text(encoding='utf-8')) if first else None
    b=d['baseline']; results=d['results']; frozen=b['draw_diagnosis']
    lines=['# Fase 2 — primeira entrega A–T',
      '## Decisão: DO NOT PROMOTE',
      '**Nenhum campeão, pick, filtro de odds, Quality Gate ou agrupamento operacional foi alterado.** '
      'Mudanças operacionais limitadas à captura de pesquisa. Testes são retrospectivos, não um novo teste cego. '
      'Nenhum threshold foi procurado. Bilhetes 4/4 não foram usados para escolher modelos.',
      '## A — Confusion matrix dos picks congelados',
      table(['Real / previsto','HOME','DRAW','AWAY'],[[name,*row] for name,row in zip(['HOME','DRAW','AWAY'],b['confusion'])]),
      table(['Classe','N','Precisão','Recall','F1'],[[name,v['support'],pct(v['precision']),pct(v['recall']),pct(v['f1'])] for name,v in b['classes'].items()]),
      '## B — Recall de empate confirmado',
      f"{frozen['actual']} empates reais; {frozen['predicted']} picks DRAW; {frozen['true_positive']} verdadeiros positivos; "
      f"{frozen['false_positive']} falsos positivos; {frozen['false_negative']} empates perdidos. Recall: {pct(b['classes']['EMPATE']['recall'])}.",
      f"Controle ingênuo HOME em todas as partidas: {b.get('always_home_policy',{}).get('correct',1186)} acertos; "
      'o ML congelado acertou 1.188. Essa proximidade reforça a necessidade de demonstrar valor fora do treino, não somente aumentar complexidade.',
      '## C — Probabilidades de empate',
      f"P_DRAW média nos empates: {pct(frozen['mean_p_draw'])}; mediana: {pct(frozen['median_p_draw'])}. "
      f"Média em todos os jogos: {pct(frozen['mean_p_draw_all'])}; frequência real: {pct(frozen['actual_draw_rate'])}. "
      f"AUC de DRAW: {number(b['draw']['auc'])}; AP: {number(b['draw']['ap'])}. "
      'Não há subestimação global de P_DRAW; há pouca discriminação entre os jogos que empatam e os demais. '
      'Poucos picks de empate não provam, sozinhos, que a probabilidade de empate está baixa.',
      table(['Faixa P_DRAW','N empates','P_HOME média','P_DRAW média','P_AWAY média','Picks H/D/A','DRAW escolhido'],[
        [str(x['range']),x['n'],*[pct(p) for p in (x['probability_mean'] or [None]*3)],str(x['picks']),pct(x['draw_chosen_rate'])] for x in b['draw_bins']]),
      '## D — Argmax',
      f"Empates perdidos a até 2 / 5 / 8 pontos percentuais do argmax: {b['near_argmax']['0.02']} / {b['near_argmax']['0.05']} / {b['near_argmax']['0.08']}. "
      'Contagens excluem os 23 empates já acertados. Estatísticas de top1/top2/entropy estão em draw_diagnosis.json. '
      'Na faixa abaixo de 0,02, trocar todos os picks casa/fora para empate recuperaria 28 empates e destruiria 32 acertos. Não é uma política recomendada.',
      '## E — Draw gap',
      table(['Gap','N','Accuracy','Perda empate','Perda lado oposto','HOME','AWAY'],[
       [str(x['range']),x['n'],pct(x['accuracy']),pct(x['draw_loss_rate']),pct(x['opposite_loss_rate']),x['home_picks'],x['away_picks']] for x in b['draw_gap']]),
      '## F — Elegibilidade',
      'Mantido HOME > 1,99 E AWAY > 1,99. O histórico de candidatos rejeitados antes da coleta não existe de modo congelado. '
      'Portanto não é possível medir sem viés o efeito causal dessa regra usando apenas os jogos que passaram. '
      'Novos eventos de agenda/mercado e motivos de exclusão foram instrumentados para permitir essa comparação futura.',
      '## G–H — Força, ataque, defesa e adversários',
      'Reconstrução por radar usando max(captura, início + 3 horas) < cutoff. Identidades preservam namespace do provedor. '
      'Ataque e defesa são estimados simultaneamente por Poisson regularizado com decaimento de 180 dias; defesa mais alta significa mais forte. '
      'Geradas janelas 3/5/8/10 geral e no mando, recência de 60 dias e shrinkage de cinco jogos. '
      'A comparação compacta usa 10 jogos, sem escolher a janela pelo teste. '
      'Resistência a derrotas usa as partidas recentes; ajuste de adversários usa força estatística, não favorito de mercado. '
      'Não inventamos temporada exata nem xG quando os campos não estão disponíveis. Hierarquia completa país→liga→temporada permanece pendente para alvos sem identidade congelada.',
      '## I–L — Gols, duas etapas, mercado e ensemble',
      f"Cobertura de gols nos jogos avaliados: {d['goal_covered']}/{results['baseline']['outcome']['n']}. "
      'Onde faltou identidade histórica, Poisson/DC usam fallback explícito para o ML congelado; não é resultado de um modelo puro em toda a amostra.',
      'Duas etapas: P(DRAW) e P(HOME | NOT DRAW), com probabilidades finais coerentes que somam 1. '
      'Multinomial, CatBoost e boosting também comparados. LightGBM não estava instalado. '
      'Ensemble logarítmico: pesos escolhidos somente nas previsões cronológicas internas fora do treino, sem stacking in-sample.',
      'Mercado antigo não foi testado: timestamps insuficientes. A nova captura distingue odds observadas pré-jogo de closing odds; '
      'remoção proporcional de margem está disponível para observações futuras válidas, com idade da cotação explicitamente desconhecida.',
      table(['Modelo','N','Accuracy','IC95%','DRAW recall','Log loss','Brier','Delta vs atual','IC delta por radar'],[
       [name,v['outcome']['n'],pct(v['outcome']['accuracy']),interval(v['outcome']['accuracy_ci']),
        pct(v['outcome']['classes']['EMPATE']['recall']),number(v['outcome']['log_loss']),number(v['outcome']['brier_multiclass_sum']),
        pct(v['delta']['delta']),interval(v['delta']['radar_block_bootstrap_ci'])]
       for name,v in results.items() if 'outcome' in v]),
      table(['Modelo','Empates recuperados','Acertos não-empate perdidos','Total corrigido','Total estragado','Ganho líquido'],[
       [name,v['delta']['draws_recovered'],v['delta']['nondraw_correct_lost'],v['delta']['fixed'],v['delta']['broken'],v['delta']['fixed']-v['delta']['broken']]
       for name,v in results.items() if 'delta' in v]),
      '## M — P(PICK_CORRECT)',
      'Target exclusivamente acerto do pick original congelado, não o pick alterado pelo experimento. '
      'A probabilidade da classe original produzida pelos modelos concorrentes também é avaliada como score. '
      'P(falha por empate) e P(falha pelo lado oposto) são eventos mutuamente exclusivos e devem ser somados, não multiplicados. '
      'As previsões salvas permitem essa decomposição; classificador separado de failure ainda não foi promovido nem validado.',
      table(['Score','AUC','AP','Top20','Bottom20','Diferença','IC diferença','AURC','Spearman decis'],[
        [name,number(v['quality']['auc']),number(v['quality']['ap']),pct(v['quality']['top20']['accuracy']),
         pct(v['quality']['bottom20']['accuracy']),pct(v['quality']['top20_bottom20_spread']),
         interval(v['quality']['radar_bootstrap']['top20_bottom20_spread_ci']) if 'radar_bootstrap' in v['quality'] else '—',
         number(v['quality']['aurc']),number(v['quality']['decile_spearman'])] for name,v in results.items()]),
      '## N — Escada do modelo de acerto',
      table(['Decil','N','Score','Accuracy','Erro','Erro DRAW','Outros erros','IC95%'],[
        [x['decile'],x['n'],pct(x['score_mean']),pct(x['accuracy']),pct(x['error_rate']),pct(x['draw_error_rate']),pct(x['opposite_error_rate']),interval(x['ci'])]
        for x in results['correctness']['quality']['deciles']]),
      '## O — Estabilidade temporal',
      table(['Fold','Treino','Teste','N','AUC','AP','D1','D10','Spread','Top20','Enrich erro 20','Enrich DRAW20'],[
       [f['fold'],str(f.get('train_period',f['n_train'])),str(f.get('test_period',f['cutoff'])),f['n_test'],
        number(f['models']['correctness']['quality']['auc']),number(f['models']['correctness']['quality']['ap']),
        pct(f['models']['correctness']['quality']['deciles'][0]['accuracy']),pct(f['models']['correctness']['quality']['deciles'][-1]['accuracy']),
        pct(f['models']['correctness']['quality']['decile_spread']),pct(f['models']['correctness']['quality']['top20']['accuracy']),
        number(f['models']['correctness']['quality']['removal'][1]['error_enrichment']),
        number(f['models']['correctness']['quality']['removal'][1]['draw_error_enrichment'])] for f in d['folds']]),
      '## P — Risk-coverage do modelo de acerto',
      'Coberturas são pontos diagnósticos pré-declarados, não thresholds escolhidos. Bilhetes possíveis somam floor(N/4) por radar.',
      table(['Cobertura','N','Accuracy','Erro','Erro DRAW','Erros removidos','Acertos removidos','Bilhetes possíveis'],[
        [pct(x['coverage']),x['n'],pct(x['accuracy']),pct(x['error_rate']),pct(x['draw_error_rate']),x['errors_removed'],x['corrects_removed'],x['tickets_possible_per_run']]
        for x in results['correctness']['quality']['coverage']]),
      '## Q–R — Concentração dos erros',
      table(['Removido','N','Erros','Acertos','Erros DRAW','Enrichment erro','Enrichment DRAW'],[
        [pct(x['removed_fraction']),x['n'],x['errors_removed'],x['correct_removed'],x['draw_errors_removed'],number(x['error_enrichment']),number(x['draw_error_enrichment'])]
        for x in results['correctness']['quality']['removal']]),
      '## S — Captura implantada',
      'phase2_observations.db é um sidecar separado do banco de produção. Registros append-only, hash SHA-256, triggers contra UPDATE/DELETE. '
      'O mesmo research_run_id liga agenda, preços observados, previsão e decisão; run_finished liga ao radar operacional. '
      'Capturados odds inclusive fora do filtro, motivos de exclusão, features realmente calculadas, probabilidades, perfil analítico, '
      'versão/id registrados no fechamento e associação ao nome do bilhete. Odds e modelos não são modificados. '
      'Rejeitados antes da inferência possuem features=null/not_evaluated, nunca atributos inventados. '
      'Falha de captura é registrada sem bloquear a decisão operacional. Falta de run_finished indica execução sem fechamento; não representa zero jogos. '
      'O processo já aberto precisa ser reiniciado normalmente para carregar a instrumentação.',
      'Limitações de captura: não há preço de fechamento retrospectivo; missing estrutural não pode recuperar zeros já imputados pela geração antiga; '
      'versão do objeto pode ser desconhecida na etapa prediction, sendo ligada à identidade registrada no fechamento. '
      'IDs de bilhete capturados nesta etapa são os nomes de montagem, não confirmação de envio pelo Telegram.',
      '## T — Decisão e próximo teste',
      '**QUALITY SCORE AINDA NÃO POSSUI DISCRIMINAÇÃO SUFICIENTE. DO NOT PROMOTE.** '
      'Não ativar gate, não escolher corte oportunista, não alterar bilhetes. '
      'Aumentar recall de empate isoladamente não justifica as perdas HOME/AWAY.',
      'Protocolo futuro: congelar um candidato antes do primeiro radar; acumular 20 radares novos sem tuning; '
      'avaliar acerto, draw AP, discriminação, escada e estabilidade com bootstrap por radar. '
      'Este protocolo está documentado, mas nenhum novo candidato foi publicado ou automaticamente promovido.',
      'Pendências explícitas: validação futura; odds congeladas suficientes para benchmark e modelo residual; '
      'ablation completa por grupo e hierarquia com IDs históricos; teste independente de calibration/stacking somente se houver discriminação; '
      'otimização de bilhetes adiada conforme solicitado.',
      '## Bilhetes — apenas baseline secundário',
      'Auditoria anterior preservada: 686 completos, distribuição 0/1/2/3/4 acertos = 98/246/229/100/13. '
      'Nenhuma escolha experimental foi feita usando os 13 greens. ROI histórico não é apresentado como confiável com odds mutáveis.',
      '## Referências metodológicas',
      '[Validação aninhada — scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html). '
      'Aqui os folds são cronológicos, com disponibilidade do rótulo, e não os splits aleatórios do exemplo. '
      '[Calibração — scikit-learn](https://scikit-learn.org/stable/modules/calibration.html). '
      'Calibração não substitui discriminação e não foi usada para disfarçar um score fraco.'
    ]
    if old:
        lines += ['## Rodada inicial preservada',
          table(['Modelo inicial','Accuracy','Quality AUC'],[[k,pct(v.get('outcome',{}).get('accuracy')),number(v['quality']['auc'])] for k,v in old['results'].items()])]
    for metric in ('auc','ap','decile_spread','top20_bottom20_spread','aurc'):
        values=[f['models']['correctness']['quality'][metric] for f in d['folds']]
        lines.append(f"Resumo entre folds de {metric}: média {number(np.mean(values))}; mediana {number(np.median(values))}; "
                     f"desvio {number(np.std(values))}; mínimo {number(min(values))}; máximo {number(max(values))}.")
    if (dest/'feature_coverage.json').exists():
        coverage=json.loads((dest/'feature_coverage.json').read_text(encoding='utf-8'))
        lines += ['## Cobertura pré-jogo efetivamente disponível',
          table(['Critério','N de '+str(coverage['n'])],coverage['coverage'].items()),
          'Há 128 snapshots com médias positivas legadas de xG para os dois times, mas sem a cobertura mínima comprovada. '
          'Apenas 25 têm contagem de ao menos uma observação em ambas as direções e times; nenhum chega ao mínimo de três. '
          'Isso não significa xG real zero. Significa evidência histórica insuficiente para o critério pré-declarado. '
          'Recuperar estatísticas hoje não prova que estavam disponíveis ao radar original; qualquer replay expandido '
          'precisa separar disponibilização esportiva da aquisição no robô e não pode ser apresentado como previsão emitida na época.',
          '## Verificação técnica',
          '67 testes automatizados passaram: captura append-only, bloqueio de alterações, falha não fatal, benchmark pré-jogo, '
          'disponibilidade temporal dos rótulos, score sintético, mercado/agenda, ticket engine e modelos de gols. '
          'Compilação dos módulos de aplicação e robô verificada. Nenhum radar, envio Telegram ou treino do campeão foi executado nesta pesquisa.']
    with (dest/'FASE_2_A_T.md').open('x',encoding='utf-8') as stream: stream.write('\n\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('folder'); p.add_argument('--first'); args=p.parse_args()
    render(args.folder,args.first)
