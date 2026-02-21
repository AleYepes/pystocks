import pandas as pd

df=pd.read_csv('research_yields.csv')
# evaluate current gating policy implied by fundamentals.py
# profile always
# holdings+ratios if has_top10 or has_objective
# lipper if overall_ratings
# dividends if has_dividends
# mstar if has_mstar or has_top10
# perf if has_perf or has_objective
# risk if has_risk or has_objective
# ownership+sentiment tick if has_ownership

m=df

def eval_gate(name, gate):
    called = gate
    yield_col='yield_'+name
    called_rate = called.mean()
    cov = m.loc[called, yield_col].mean() if called.any() else float('nan')
    miss = m.loc[~called, yield_col].mean() if (~called).any() else float('nan')
    # missed positives: endpoint would yield but not called
    missed = ((~called) & m[yield_col]).mean()
    unnecessary = (called & ~m[yield_col]).mean()
    print(name.ljust(10), 'called_rate', round(called_rate,4), 'precision', round(cov,4) if cov==cov else cov, 'missed_pos_rate', round(missed,4), 'unnecessary_rate', round(unnecessary,4), 'yield_when_not_called', round(miss,4) if miss==miss else miss)

has_top10=m['landing_top10']
has_obj=m['landing_objective']
has_over=m['landing_overall_ratings']
has_div=m['landing_dividends']
has_mstar=m['landing_mstar']
# typo in research script: cumulative_performace
has_perf=m['landing_cumulative_performace'] if 'landing_cumulative_performace' in m.columns else m['landing_top10']*False
has_risk=m['landing_risk_statistics']
has_owner=m['landing_ownership']

print('gating eval current approach')
eval_gate('holdings', has_top10 | has_obj)
eval_gate('ratios', has_top10 | has_obj)
eval_gate('lipper', has_over)
eval_gate('divs', has_div)
eval_gate('mstar', has_mstar | has_top10)
eval_gate('perf', has_perf | has_obj)
eval_gate('risk', has_risk | has_obj)
eval_gate('owner', has_owner)
def stats(gate, col='yield_mstar'):
    called = gate
    prec = m.loc[called, col].mean() if called.any() else float('nan')
    missed = ((~called) & m[col]).mean()
    call = called.mean()
    print('call', round(call, 4), 'precision', round(prec, 4), 'missed', round(missed, 4), 'yield_not_called', round(m.loc[~called, col].mean(), 4) if (~called).any() else 'nan')

print('\nmstar gating options:')
print('mstar|top10')
stats(m['landing_mstar'] | m['landing_top10'])
print('mstar|key_profile')
stats(m['landing_mstar'] | m['landing_key_profile'])
print('top10|key_profile')
stats(m['landing_top10'] | m['landing_key_profile'])
print('objective|top10')
stats(m['landing_objective'] | m['landing_top10'])
print('always')
stats(pd.Series([True] * len(m)))
