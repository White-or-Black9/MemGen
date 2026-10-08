"""M4-B1: CPU-only reconstruction of saved retrieval arithmetic and selection.

No latent decoding, query/key tensor recomputation, model imports or new inference.
"""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m4_provenance', HERE / 'audit_provenance.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
c, require = a.c, a.require
TOL = dict(rel_tol=1e-12, abs_tol=1e-12)


def rank(values, descending=True):
    require(all(math.isfinite(x) for x in values), 'nonfinite score')
    return sorted(range(len(values)), key=lambda i: (-values[i] if descending else values[i], i))


def reconstruct(cosine, ages, alpha, threshold, top_k):
    require(len(cosine) == len(ages) and all(x >= 0 for x in ages), 'invalid age/score arrays')
    decay = [math.exp(-alpha * age) for age in ages]
    scores = [v * d for v, d in zip(cosine, decay)]
    ordered = rank(scores)
    eligible = [i for i in ordered if scores[i] >= threshold]
    native = eligible[:top_k]
    n = len(native)
    return dict(decay=decay, scores=scores, native=native, full=sorted(native),
                eligible=eligible, cosine_order=rank(cosine), score_order=ordered,
                recency=sorted(rank(ages, descending=False)[:n]))


def close_list(left, right, name):
    require(len(left) == len(right) and all(math.isclose(x, y, **TOL) for x, y in zip(left, right)),
            name + ' arithmetic mismatch')


def paired(traces):
    require(len(traces) == len(c.POLICIES) and {t['policy'] for t in traces} == set(c.POLICIES),
            'missing or duplicate policies')
    first = traces[0]
    for t in traces:
        for name in ('cosine', 'ages', 'last_write', 'decay', 'final_scores', 'native_indices', 'n'):
            require(t[name] == first[name], 'paired base drift: ' + name)


def metrics_range(values):
    return dict(min=min(values), max=max(values)) if values else None


def run(campaign, provenance_run, output):
    start = time.monotonic()
    require(not output.exists(), 'choose a fresh output directory')
    require(not output.is_relative_to(campaign) and not output.is_relative_to(provenance_run),
            'output must not be inside parent evidence')
    parent_before, source_before = a.inventory(campaign), a.inventory(provenance_run)
    manifest = c.read(campaign / 'manifest.json')
    require(manifest['contract_sha256'] == a.CONTRACT, 'wrong parent campaign')
    require(c.read(campaign / 'results.json')['status'] == 'COMPLETE', 'parent not complete')
    acceptance = c.read(HERE / 'acceptance.json')
    require(acceptance['status'] == 'PASS', 'M4-A not accepted')
    for name, sha in acceptance['identical_artifact_sha256'].items():
        require(c.sha(provenance_run / name) == sha, 'M4-A artifact drift: ' + name)
    output.mkdir(parents=True)
    c.atomic(output / 'status.json', dict(status='RUNNING', started_at=c.now()))
    parent_hashes = c.read(provenance_run / 'manifest.json')['parent_json_sha256']
    for name, sha in parent_hashes.items():
        require(c.sha(campaign / name) == sha, 'parent JSON drift since M4-A: ' + name)
    audit_spec = importlib.util.spec_from_file_location('m3_audit', HERE.parent / 'm3/audit_partial.py')
    old = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(old)
    integrity = old.audit(campaign)
    require(integrity['verified_records'] == 15000 and integrity['queue_status_counts'] == {'done': 25},
            'parent incomplete')
    c.atomic(output / 'parent_integrity.json', integrity)
    provenance = {}
    for line in (provenance_run / 'provenance.jsonl').read_text().splitlines():
        row = json.loads(line)
        key = row['job_id'], row['slot_index']
        require(key not in provenance, 'duplicate slot version')
        provenance[key] = row
    protocol = manifest['protocol']
    alpha, threshold, top_k = (protocol[k] for k in ('decay_alpha', 'retrieve_threshold', 'top_k'))
    diagnostics, job_rows = [], []
    for job in manifest['jobs']:
        directory = campaign / 'jobs' / job['id']
        pre = c.read(directory / 'construction.json')['pre_query_bank_summary']
        slots = [provenance[job['id'], i] for i in range(pre['slot_count'])]
        current_step = len(c.read(directory / 'construction.json')['construction_turn_diagnostics']) + 1
        frozen_ages = [max(0, current_step - s['last_retrieved_step']) for s in slots]
        write_ages = [current_step - s['last_write_step'] for s in slots]
        require(pre['write_count'] + 1 == current_step, 'write/retrieval count scale mismatch')
        frequencies, order_frequency = Counter(), Counter()
        cosine_vectors, query_vectors, cosine_orders, score_orders = set(), set(), set(), set()
        eligible_counts, exact, overlap = Counter(), Counter(), Counter()
        cosine_values, final_values, margins, component_errors = [], [], [], []
        ns, age_ties = Counter(), Counter()
        for q in range(c.QUESTIONS):
            envelopes = [c.read(directory / 'queries' / f'q{q:03d}_{p}.json') for p in c.POLICIES]
            traces = [e['record']['selection'][0] for e in envelopes]
            paired(traces)
            by_policy = {t['policy']: t for t in traces}
            full = by_policy['full']
            require(full['ages'] == frozen_ages, 'query access ages changed from frozen construction state')
            require(full['last_write'] == [s['last_write_step'] for s in slots], 'write provenance drift')
            calc = reconstruct(full['cosine'], frozen_ages, alpha, threshold, top_k)
            for t, envelope in zip(traces, envelopes):
                close_list(t['decay'], calc['decay'], 'decay')
                close_list(t['final_scores'], calc['scores'], 'score')
                require(t['native_indices'] == calc['native'], 'native ranking mismatch')
                require(t['n'] == len(calc['native']), 'count mismatch')
                policy = t['policy']
                n = t['n']
                expected = {'full': calc['full'], 'native_order': calc['native'],
                            'recency_only': calc['recency'],
                            'cosine_only': sorted(calc['cosine_order'][:n]),
                            'last_written': sorted(rank(full['last_write'])[:n]),
                            'random': sorted(random.Random(t['random_seed']).sample(range(len(slots)), n))}[policy]
                require(t['selected_indices'] == expected, 'saved selector mismatch: ' + policy)
                decomp = envelope['runtime_result']['query_score_decomposition']
                require(decomp['retrieval_step'] == current_step, 'retrieval step drift across queries')
                require(len(decomp['slots']) == len(slots), 'missing score decomposition slots')
                for i, d in enumerate(decomp['slots']):
                    require(d['slot_index'] == i and d['last_retrieved_age'] == frozen_ages[i], 'decomposition index/age drift')
                    require(d['raw_cosine'] == full['cosine'][i], 'native/adaptor raw cosine mismatch')
                    require(d['threshold_passed'] == (calc['scores'][i] >= threshold), 'threshold boundary mismatch')
                    require(d['raw_cosine_rank'] == calc['cosine_order'].index(i) + 1 and
                            d['final_score_rank'] == calc['score_order'].index(i) + 1, 'saved rank mismatch')
                    close_list([d['decay_factor'], d['final_score']], [calc['decay'][i], calc['scores'][i]], 'decomposition')
                require(decomp['query_vector_hash'] == envelopes[0]['runtime_result']['query_score_decomposition']['query_vector_hash'],
                        'query representation drift across policies')
            chosen = calc['full']
            ns[len(chosen)] += 1
            frequencies[tuple(chosen)] += 1
            order_frequency[tuple(calc['native'])] += 1
            cosine_vectors.add(tuple(full['cosine']))
            query_vectors.add(envelopes[0]['runtime_result']['query_score_decomposition']['query_vector_hash'])
            cosine_orders.add(tuple(calc['cosine_order']))
            score_orders.add(tuple(calc['score_order']))
            eligible_counts[len(calc['eligible'])] += 1
            cosine_values.extend(full['cosine'])
            final_values.extend(calc['scores'])
            component_errors.extend(abs(v - w) for v, w in zip(full['final_scores'], calc['scores']))
            for policy in ('recency_only', 'last_written', 'cosine_only'):
                other = by_policy[policy]['selected_indices']
                exact[policy] += chosen == other
                overlap[policy] += len(set(chosen) & set(other))
            age_cutoff = sorted(frozen_ages)[len(chosen) - 1] if chosen else None
            tied = [i for i, age in enumerate(frozen_ages) if age == age_cutoff] if chosen else []
            age_ties[len(tied)] += 1
            rejected = [i for i in range(len(slots)) if i not in chosen]
            margin = min(calc['scores'][i] for i in chosen) - max(calc['scores'][i] for i in rejected) if chosen and rejected else None
            if margin is not None:
                margins.append(margin)
            diagnostics.append(dict(job_id=job['id'], seed=job['seed'], context_index=job['context_index'], query_id=q,
                                    n=len(chosen), full=chosen, native_order=calc['native'],
                                    recency=calc['recency'], last_written=by_policy['last_written']['selected_indices'],
                                    cosine_only=by_policy['cosine_only']['selected_indices'],
                                    cosine=full['cosine'], access_ages=frozen_ages, write_ages=write_ages,
                                    decay=calc['decay'], final_scores=calc['scores'], threshold_eligible=calc['eligible'],
                                    cosine_order=calc['cosine_order'], score_order=calc['score_order'],
                                    selection_margin=margin, recency_cutoff_tie_slots=tied))
        selected_counts = Counter(i for indices, count in frequencies.items() for i in indices for _ in range(count))
        slot_rows = [dict(slot_index=i, current_write_chunk=s['direct_source_chunk'],
                          write_age=write_ages[i], access_age=frozen_ages[i], construction_access_count=s['access_count'],
                          selected_queries=selected_counts[i], slot_version=s['slot_version']) for i, s in enumerate(slots)]
        job_rows.append(dict(job_id=job['id'], seed=job['seed'], context_index=job['context_index'], queries=100,
                             n_counts=dict(ns), full_unique_sets=len(frequencies),
                             full_set_counts={','.join(map(str, key)): v for key, v in frequencies.items()},
                             native_order_unique=len(order_frequency), cosine_vectors_unique=len(cosine_vectors),
                             query_vector_hashes_unique=len(query_vectors), cosine_rankings_unique=len(cosine_orders),
                             final_rankings_unique=len(score_orders), threshold_eligible_counts=dict(eligible_counts),
                             exact_set_match_counts=dict(exact), intersection_slot_counts=dict(overlap),
                             recency_cutoff_tie_size_counts=dict(age_ties), cosine_range=metrics_range(cosine_values),
                             final_score_range=metrics_range(final_values), selection_margin_range=metrics_range(margins),
                             max_arithmetic_abs_error=max(component_errors), slots=slot_rows,
                             caveat='source chunk age is not gold evidence age; repeated queries restore frozen metadata'))
    require(len(diagnostics) == 2500 and len({(d['job_id'], d['query_id']) for d in diagnostics}) == 2500,
            'duplicate/missing seed-question diagnostics')
    require('torch' not in sys.modules, 'unexpected Torch import')
    require(time.monotonic() - start < 1800, '30-minute budget exceeded')
    a.check_unchanged(parent_before, campaign)
    a.check_unchanged(source_before, provenance_run)
    a.jsonl(output / 'diagnostics.jsonl', diagnostics)
    c.atomic(output / 'by_job.json', job_rows)
    summary = dict(status='COMPLETE', jobs=25, policy_records_verified=15000, seed_question_groups=2500,
                   unique_questions=500, independent_contexts=5, source_semantic_labels='unknown; not analyzed',
                   n_counts=dict(sum((Counter({int(k): v for k, v in j['n_counts'].items()}) for j in job_rows), Counter())),
                   jobs_with_constant_full_set=sum(j['full_unique_sets'] == 1 for j in job_rows),
                   jobs_with_query_varying_cosine=sum(j['cosine_vectors_unique'] > 1 for j in job_rows),
                   exact_set_match_counts={p: sum(j['exact_set_match_counts'][p] for j in job_rows)
                                           for p in ('recency_only', 'last_written', 'cosine_only')},
                   max_arithmetic_abs_error=max(j['max_arithmetic_abs_error'] for j in job_rows),
                   contexts=[dict(context_index=i, job_ids=[j['job_id'] for j in job_rows if j['context_index'] == i],
                                  full_unique_sets=[j['full_unique_sets'] for j in job_rows if j['context_index'] == i]) for i in range(5)],
                   parent_preserved=True, provenance_parent_preserved=True,
                   query_metadata_restored=True, torch_loaded=False,
                   next_route='plan B2 paired candidate-validity and strict-correctness decomposition; no redesign or new inference')
    c.atomic(output / 'summary.json', summary)
    c.atomic(output / 'exceptions.json', dict(arithmetic_errors=[], missing_records=[], unknowns=[
        'Cosine tensors cannot be recomputed from JSON; saved native/adaptor raw scores cross-checked.',
        'No semantic source labels; no Recall/MRR or evidence-age claim.',
        'No causal ablation of construction feedback or decay.']))
    c.atomic(output / 'manifest.json', dict(created_at=c.now(), parent_contract_sha256=a.CONTRACT,
             analyzer_sha256=c.sha(__file__), provenance_analyzer_sha256=c.sha(HERE / 'audit_provenance.py'),
             parent_manifest_sha256=c.sha(campaign / 'manifest.json'), parent_integrity=integrity['verified_records'],
             parent_json_hashes_verified_against_m4a=True, parent_json_hash_count=len(parent_hashes),
             provenance_inputs={name: c.sha(provenance_run / name) for name in acceptance['identical_artifact_sha256']},
             source_input_hashes_verified=True, tolerance=TOL, comparison_baselines=list(c.POLICIES),
             exp_id='M4-B1', item_id='M4-B1', section_id='retrieval_diagnostics', paper_role='main_text',
             claim_links=['R03', 'R04', 'R07'], evidence_role='claim-carrying limitation analysis',
             selected_outline_ref='EXPERIMENT_PLAN.md', research_questions=['RQ-M4'],
             experimental_designs=['offline trace reconstruction'], todo_items=['B1'],
             python=sys.executable, python_version=platform.python_version(),
             resource_contract='single CPU process; no Torch; 30 minutes; output <1GiB',
             comparability='same frozen M3 outputs; no model or inference changes', elapsed_seconds=time.monotonic() - start))
    a.check_unchanged(parent_before, campaign)
    a.check_unchanged(source_before, provenance_run)
    c.atomic(output / 'status.json', dict(status='COMPLETE', finished_at=c.now()))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--provenance', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dest = Path(args.output).resolve()
    existed = dest.exists()
    try:
        run(Path(args.campaign).resolve(), Path(args.provenance).resolve(), dest)
    except Exception as exc:
        if not existed and (dest / 'status.json').exists():
            c.atomic(dest / 'status.json', dict(status='FAILED', error=f'{type(exc).__name__}: {exc}'))
        raise
