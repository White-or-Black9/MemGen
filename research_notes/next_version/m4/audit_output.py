"""M4-B2: offline paired output states; no causal mediation or semantic labels."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m4_sources', HERE / 'audit_provenance.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
c, require = a.c, a.require
from scripts.eval import review_eventqa_decomposition as m

STATES = ('correct_valid', 'wrong_valid', 'invalid')


def state(row):
    require(not row['strict_candidate_correct'] or row['candidate_valid'], 'correct but invalid candidate')
    return 'correct_valid' if row['strict_candidate_correct'] else 'wrong_valid' if row['candidate_valid'] else 'invalid'


def invalid_type(row, info, official):
    if row['candidate_valid']:
        return None
    if row['parser_none']:
        return 'parser_none'
    if row['parser_empty']:
        return 'parser_empty'
    if row['candidate_match_count'] > 1:
        return 'ambiguous_normalization'
    parsed = official.normalize_answer(row['parsed'])
    # This subtype is diagnostic, not a change to the frozen valid-candidate rule.
    if sum(bool(candidate) and candidate in parsed for candidate in set(info['normalized'])) > 1:
        return 'multiple_candidate_text'
    return 'non_candidate'


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = row['policy'], row['seed'], row['context_index'], row['query_id']
        require(key not in indexed, 'duplicate scored identity')
        indexed[key] = row
    return indexed


def aligned(full, other):
    for field in ('seed', 'context_index', 'query_id', 'context_id', 'question', 'gold_answers', 'query_sha256'):
        require(full[field] == other[field], 'paired identity mismatch: ' + field)


def transitions(pairs):
    table = {s: {t: 0 for t in STATES} for s in STATES}
    for full, other in pairs:
        aligned(full, other)
        table[state(other)][state(full)] += 1
    n = len(pairs)
    full_marginal = {s: sum(table[t][s] for t in STATES) for s in STATES}
    other_marginal = {s: sum(table[s].values()) for s in STATES}
    require(sum(full_marginal.values()) == sum(other_marginal.values()) == n, 'nonexhaustive table')
    wins = sum(table[s]['correct_valid'] for s in STATES if s != 'correct_valid')
    losses = sum(table['correct_valid'][s] for s in STATES if s != 'correct_valid')
    invalid_net = table['invalid']['correct_valid'] - table['correct_valid']['invalid']
    valid_wrong_net = table['wrong_valid']['correct_valid'] - table['correct_valid']['wrong_valid']
    net = full_marginal['correct_valid'] - other_marginal['correct_valid']
    require(net == wins - losses == invalid_net + valid_wrong_net, 'transition net not conserved')
    common = [(f, o) for f, o in pairs if f['candidate_valid'] and o['candidate_valid']]
    return dict(count=n, table_other_to_full=table, full_states=full_marginal, other_states=other_marginal,
                wins=wins, losses=losses, ties=n-wins-losses, net_correct=net,
                delta_strict=net/n if n else None, invalid_correct_net=invalid_net,
                valid_wrong_correct_net=valid_wrong_net,
                common_valid=dict(denominator=len(common), full=m.fraction(sum(f['strict_candidate_correct'] for f,o in common),len(common)),
                                  other=m.fraction(sum(o['strict_candidate_correct'] for f,o in common),len(common)),
                                  caveat='output-selected subset; selection bias; not causal evidence'),
                equality_counts={field: sum(f[field] == o[field] for f,o in pairs)
                                 for field in ('prediction', 'parsed', 'candidate_valid', 'strict_candidate_correct', 'selected_candidate')})


def run(parent, output):
    start = time.monotonic()
    require(not output.exists() and not output.is_relative_to(parent), 'choose a fresh non-parent output')
    before = a.inventory(parent)
    manifest = c.read(parent / 'manifest.json')
    require(manifest['contract_sha256'] == a.CONTRACT, 'wrong campaign')
    require(c.read(HERE / 'acceptance_b1.json')['status'] == 'PASS', 'B1 not accepted')
    frozen = c.read(HERE / 'run-20261008-a1/manifest.json')['parent_json_sha256']
    output.mkdir(parents=True)
    c.atomic(output / 'status.json', dict(status='RUNNING', started_at=c.now()))
    for name, sha in frozen.items():
        require(c.sha(parent / name) == sha, 'parent hash drift: ' + name)
    c.verify_manifest(manifest, check_inputs=True)
    require(all(j['status'] == 'done' for j in c.read(parent/'queue.json')['jobs'].values()), 'unfinished parent jobs')
    result = c.read(parent / 'results.json')
    require(result['status'] == 'COMPLETE' and result['count'] == 15000, 'parent score incomplete')
    rows = c.read(parent / 'scored_records.json')
    indexed = index_rows(rows)
    expected = {(p,j['seed'],j['context_index'],q) for j in manifest['jobs'] for q in range(100) for p in c.POLICIES}
    require(set(indexed) == expected, 'missing/unexpected scored records')
    official = m.load_official(c.MAB / 'utils/eval_other_utils.py')
    infos, states, compact = {}, {}, []
    for key, row in indexed.items():
        _, seed, ctx, q = key
        base = seed, ctx, q
        if base not in infos:
            infos[base] = m.candidate_info(row['question'], row['gold_answers'], official)
        info = infos[base]
        require(not info['errors'], 'candidate schema errors')
        new = m.evaluate(row['prediction'], row['gold_answers'], info, row['format_flags'], official)
        require(all(row[k] == v for k,v in new.items()), 'saved scorer field drift')
        envelope = c.read(parent / 'jobs' / f's{seed}-c{ctx}' / 'queries' / f'q{q:03d}_{row["policy"]}.json')
        require(all(row[k] == v for k,v in envelope['record'].items()), 'scored/raw record drift')
        row['selected_candidate'] = info['candidates'][info['normalized'].index(official.normalize_answer(row['parsed']))] if row['candidate_valid'] else None
        row['output_state'] = state(row)
        row['invalid_subtype'] = invalid_type(row, info, official)
        compact.append({k: row[k] for k in ('policy','seed','context_index','query_id','output_state','invalid_subtype',
                        'parsed','selected_candidate','candidate_valid','strict_candidate_correct','raw_gold_any','official_em')})
    policy_summary = {}
    for policy in c.POLICIES:
        group = [r for r in rows if r['policy'] == policy]
        summary = m.summarize(group)
        require(summary == result['summary'][policy], 'summary drift: ' + policy)
        policy_summary[policy] = dict(states=dict(Counter(r['output_state'] for r in group)),
                                      invalid_subtypes=dict(Counter(r['invalid_subtype'] for r in group if not r['candidate_valid'])),
                                      candidate_valid=summary['candidate_valid_rate'], strict=summary['strict_candidate_accuracy'],
                                      gold_mentioned_invalid=sum(r['raw_gold_any'] and not r['candidate_valid'] for r in group))
    comparisons, examples = {}, []
    for policy in c.POLICIES:
        if policy == 'full':
            continue
        pairs = [(indexed['full',j['seed'],j['context_index'],q], indexed[policy,j['seed'],j['context_index'],q])
                 for j in manifest['jobs'] for q in range(100)]
        aggregate = transitions(pairs)
        old = result['paired_full'][policy]
        require((aggregate['wins'],aggregate['losses'],aggregate['ties']) == (old['wins'],old['losses'],old['ties']), 'win/loss drift')
        require(abs(aggregate['delta_strict']-old['mean_delta']) < 1e-12, 'M3 accuracy delta drift')
        partitions = {}
        for field in ('context_index','seed'):
            partitions[field] = {str(v):transitions([(f,o) for f,o in pairs if f[field]==v]) for v in sorted({f[field] for f,o in pairs})}
        partitions['seed_context'] = {j['id']:transitions([(f,o) for f,o in pairs if f['seed']==j['seed'] and f['context_index']==j['context_index']]) for j in manifest['jobs']}
        comparisons[policy] = dict(aggregate=aggregate, partitions=partitions)
        buckets = defaultdict(list)
        for f,o in pairs:
            buckets[state(o),state(f)].append((f,o))
        for bucket, group in sorted(buckets.items()):
            picked = sorted(group, key=lambda pair:a.text_sha(f"B2|{policy}|{pair[0]['seed']}|{pair[0]['context_index']}|{pair[0]['query_id']}"))[:2]
            for f,o in picked:
                examples.append(dict(comparator=policy, transition=list(bucket), bucket_denominator=len(group),
                                     seed=f['seed'], context_index=f['context_index'], query_id=f['query_id'],
                                     full_output=f['prediction'], other_output=o['prediction']))
    require('torch' not in sys.modules, 'unexpected Torch import')
    require(time.monotonic()-start < 1800, '30-minute budget exceeded')
    a.check_unchanged(before,parent)
    a.jsonl(output/'states.jsonl',compact)
    c.atomic(output/'comparisons.json',comparisons)
    c.atomic(output/'policy_summary.json',policy_summary)
    a.jsonl(output/'examples.jsonl',examples)
    c.atomic(output/'summary.json',dict(status='COMPLETE',records=15000,seed_question_groups=2500,unique_questions=500,
             independent_contexts=5,comparisons={p:x['aggregate'] for p,x in comparisons.items()},
             parent_preserved=True,scorer_recomputed_identically=True,torch_loaded=False,
             next_route='plan claim-boundary integration / revision decision; no automatic paper edits or retrieval redesign'))
    c.atomic(output/'manifest.json',dict(created_at=c.now(),analyzer_sha256=c.sha(__file__),parent_contract_sha256=a.CONTRACT,
             parent_json_hash_count=len(frozen),inputs={n:c.sha(parent/n) for n in ['manifest.json','results.json','scored_records.json']},
             scorer_sha256=c.sha(a.ROOT/'scripts/eval/review_eventqa_decomposition.py'),
             official_sha256=c.sha(c.MAB/'utils/eval_other_utils.py'),
             b1_acceptance_sha256=c.sha(HERE/'acceptance_b1.json'),tolerance_for_summary_delta=1e-12,
             invalid_subtype_rule='parser None/empty then normalization ambiguity then multiple normalized candidate mentions then noncandidate',
             sample_rule='two smallest sha256(B2|policy|seed|context|query) per occupied transition bucket after full tables',
             exp_id='M4-B2',item_id='M4-B2',section_id='evaluation_decomposition',paper_role='appendix supporting',
             claim_links=['R01','R02','R03','R05'],selected_outline_ref='EXPERIMENT_PLAN.md',
             research_questions=['output-validity-versus-correctness'],experimental_designs=['offline paired decomposition'],todo_items=['B2'],
             comparability='same M3 evaluation contract; common-valid subset has selection bias; no causal mediation',
             python=sys.executable,elapsed_seconds=time.monotonic()-start))
    a.check_unchanged(before,parent)
    c.atomic(output/'status.json',dict(status='COMPLETE',finished_at=c.now()))
    print(json.dumps({p:{k:v for k,v in x['aggregate'].items() if k not in ['table_other_to_full','equality_counts']} for p,x in comparisons.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    dest=Path(args.output).resolve()
    existed=dest.exists()
    try:
        run(Path(args.campaign).resolve(),dest)
    except Exception as exc:
        if not existed and (dest/'status.json').exists():
            c.atomic(dest/'status.json',dict(status='FAILED',error=f'{type(exc).__name__}: {exc}'))
        raise
