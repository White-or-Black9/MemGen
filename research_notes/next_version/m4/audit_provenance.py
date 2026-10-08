"""M4-A offline provenance audit. No torch, model, GPU, or parent writes.

Source overlap and latent relevance are distinct nullable labels. Exact gold
matches are lexical diagnostics only; QA identifiers are not source offsets.
"""
import argparse
from collections import Counter
import importlib.util
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.eval import review_m3_formal as c

RUBRIC = 'M4-A-source-audit-v1'
CONTRACT = 'a548306d4131e4df0cba1340d3272a4882feae1d9c993c881e281f4551e433c9'


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def inventory(root):
    return {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(root.rglob('*')) if p.is_file()}


def check_unchanged(before, root):
    require(before == inventory(root), 'parent files changed during analysis')


def verify_chunks(chunks, expected):
    import hashlib
    require([hashlib.sha256(s.encode()).hexdigest() for s in chunks] == expected,
            'reconstructed chunk hash drift')


def chunk_spans(text, chunks, sentences):
    """Align the official join-of-sentences chunks to original character offsets."""
    spans, cursor = [], 0
    for sentence in sentences:
        start = text.find(sentence, cursor)
        require(start >= cursor, 'sentence not found in original context')
        spans.append([start, start + len(sentence)])
        cursor = start + len(sentence)
    result, pos = [], 0
    for index, chunk in enumerate(chunks):
        begin, assembled = pos, ''
        while pos < len(sentences) and assembled != chunk:
            assembled = (assembled + ' ' if pos > begin else '') + sentences[pos]
            pos += 1
            require(len(assembled) <= len(chunk), 'chunk sentence alignment failed')
        require(assembled == chunk and pos > begin, 'empty or unaligned chunk')
        result.append(dict(chunk_index=index, char_start=spans[begin][0],
                           char_end=spans[pos - 1][1], sentence_spans=spans[begin:pos]))
    require(pos == len(sentences), 'unconsumed source sentences')
    return result


def replay(turns, summary, chunks):
    """Slot storage replaces in place. New content has a new version, not old semantics."""
    require(len(turns) == len(chunks), 'turn/chunk mismatch under prompt-only construction')
    slots, versions, events = [], Counter(), []
    for i, turn in enumerate(turns):
        require(turn['construction_turn_index'] == i, 'missing/out-of-order construction turn')
        require(turn['slot_count_before_write'] == len(slots), 'pre-write slot count mismatch')
        step = i + 1
        require(turn['write_count_after_write'] == step, 'non-single-write construction turn')
        selected = turn['retrieved_indices_before_write']
        require(len(selected) == len(set(selected)) and all(0 <= k < len(slots) for k in selected),
                'invalid construction dependency index')
        dependencies = [slots[k]['slot_version'] for k in selected]
        indirect = sorted({j for k in selected for j in slots[k]['potential_source_chunks']})
        for k in selected:
            slots[k]['access_count'] += 1
            slots[k]['last_retrieved_step'] = step
        target, action = turn['target_slot_index'], turn['write_action']
        require(action in {'insert', 'replace_matched', 'evict_oldest_insert'}, 'unsupported write action')
        previous = None
        if action == 'insert':
            require(target == len(slots), 'insert did not append')
            refresh, thread = 0, f'thread-at-turn-{i}'
        else:
            require(target in range(len(slots)), 'replacement index outside bank')
            previous = slots[target]
            key = 'replaced_slot_index' if action == 'replace_matched' else 'evicted_slot_index'
            require(turn[key] == target, 'replacement/eviction identity mismatch')
            refresh = previous['refresh_count'] + 1 if action == 'replace_matched' else 0
            thread = previous['thread_id'] if action == 'replace_matched' else f'thread-at-turn-{i}'
        versions[target] += 1
        value = dict(slot_index=target, slot_version=f'slot{target}-v{versions[target]}',
                     thread_id=thread, refresh_count=refresh, current_write_turn=i,
                     last_write_step=step, created_step=turn['created_step_after_write'],
                     last_retrieved_step=turn['last_retrieved_step_after_write'],
                     access_count=turn['access_count_after_write'],
                     direct_source_chunk=i, direct_source_chunk_sha256=chunks[i],
                     construction_text_scope='current chunk plus system/template; no cumulative text prefix',
                     prior_retrieved_versions=dependencies, indirect_possible_source_chunks=indirect,
                     potential_source_chunks=sorted(set(indirect + [i])),
                     previous_content_version=None if previous is None else previous['slot_version'],
                     source_mapping_basis='frozen code/config: one prompt write per chunk; ordered environment',
                     latent_semantic_relevance=None,
                     semantic_unknown_reason='latent content not decoded or independently validated')
        require(value['created_step'] == value['last_retrieved_step'] == step,
                'replacement metadata does not describe a new slot')
        require(value['access_count'] == 0, 'new content access count did not reset')
        if action == 'insert':
            slots.append(value)
        else:
            slots[target] = value
        require(turn['slot_count_after_write'] == len(slots), 'post-write slot count mismatch')
        require(turn['per_slot_access_count_after_write'] == [s['access_count'] for s in slots],
                'access counter replay mismatch')
        events.append(dict(turn=i, action=action, target_version=value['slot_version'],
                           prior_retrieved_versions=dependencies,
                           superseded_version=value['previous_content_version']))
    require(summary['slot_count'] == len(slots) and summary['write_count'] == len(turns),
            'final bank summary mismatch')
    require(len(summary['slots']) == len(slots), 'incomplete final per-slot metadata')
    for actual, saved in zip(slots, summary['slots']):
        for field in ('slot_index', 'created_step', 'last_retrieved_step', 'access_count'):
            require(actual[field] == saved[field], f'final metadata drift: {field}')
        require(saved['last_retrieved_age'] == len(turns) - actual['last_retrieved_step'],
                'final age mismatch')
    return slots, events


def locate_gold(text, golds, spans):
    """All matches are lexical proxies, including unique and multi-gold matches."""
    matches = []
    for gold in dict.fromkeys(golds):
        require(isinstance(gold, str) and gold.strip(), 'invalid gold answer')
        for match in re.finditer(re.escape(gold), text, re.IGNORECASE):
            overlapped = [s['chunk_index'] for s in spans
                          if s['char_start'] < match.end() and match.start() < s['char_end']]
            matches.append(dict(gold=gold, start=match.start(), end=match.end(),
                                chunks=overlapped, crosses_chunk_boundary=len(overlapped) > 1,
                                text=text[match.start():match.end()]))
    state = 'absent' if not matches else 'unique' if len(matches) == 1 else 'multiple'
    return dict(lexical_status=state, lexical_matches=matches, source_evidence_overlap=None,
                latent_semantic_relevance=None, evidence_status='unknown_no_source_span',
                unknown_reason='QA/event summary metadata is not a verified source span; lexical matches alone insufficient')


def blind_sample(rows):
    selected = []
    for ctx in range(5):
        group = [r for r in rows if r['context_index'] == ctx]
        selected.extend(sorted(group, key=lambda r: text_sha(
            f"M4-A-20261008|{ctx}|{r['query_id']}"))[:10])
    return selected


def jsonl(path, values):
    with path.open('x', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')


def source_contract(manifest):
    """Persist reviewed source anchors; hashes are verified by the parent audit."""
    import yaml
    config = yaml.safe_load((ROOT / 'configs/latent_memory/triviaqa.yaml').read_text())
    require(config['model']['max_inference_aug_num'] == 0 and
            config['model']['trigger']['active'] is False, 'prompt-only assumption invalid')
    anchors = {
        'scripts/eval/mab5a_detectiveqa_compressed_n10.py':
            ['current_user_content = rollings["inter_histories"][0][-1]["content"]', 'rendered_messages = [['],
        'scripts/eval/mab2_bank_off.py':
            ['payload["memorization_prompts"][1:] + [payload["query_prompt"]]',
             'observation = self.subsequent_prompts[self.turn]'],
        'memgen/model/modeling_utils.py':
            ['if is_prompt:', 'over_limit = (sentence_augment_count >= max_augment_num)'],
        'memgen/model/latent_memory_bank.py': ['self._slots[matched_index] = new_slot',
                                              'self._slots[evicted_slot_index] = new_slot'],
    }
    # Trigger source filename is discovered without importing the model.
    trigger_paths = list((ROOT / 'memgen').rglob('*trigger*.py'))
    require(trigger_paths, 'trigger implementation missing')
    trigger_path = next(p for p in trigger_paths if 'class MemGenTrigger' in p.read_text())
    anchors[str(trigger_path.relative_to(ROOT))] = ['if self.active:', 'logits[..., 1] = 1.0']
    reviewed = {}
    for name, fragments in anchors.items():
        text = (ROOT / name).read_text()
        require(name in manifest['source_sha256'], 'source anchor not frozen in parent')
        for fragment in fragments:
            require(fragment in text, 'reviewed source anchor drift')
        reviewed[name] = dict(sha256=c.sha(ROOT / name), anchors=fragments)
    return dict(mapping='ordered chunk environment + compressed current observation + prompt-only writes',
                config_sha256=c.sha(ROOT / 'configs/latent_memory/triviaqa.yaml'),
                inference_augmentation_limit=0, trigger_active=False, anchors=reviewed,
                caveat='static code/config-derived linkage, not per-write recorded prompt hash')


def run(campaign, output):
    start = time.monotonic()
    require(not output.exists(), 'output exists; choose a new output directory to preserve prior audit')
    require(not output.is_relative_to(campaign) and output != campaign, 'cannot write inside parent')
    before = inventory(campaign)
    manifest = c.read(campaign / 'manifest.json')
    require(manifest['contract_sha256'] == CONTRACT, 'wrong parent campaign')
    result = c.read(campaign / 'results.json')
    require(result['status'] == 'COMPLETE' and result['count'] == 15000, 'parent not accepted')
    output.mkdir(parents=True)
    c.atomic(output / 'status.json', dict(status='RUNNING', started_at=c.now()))
    spec = importlib.util.spec_from_file_location('m3_prefix_audit', ROOT / 'research_notes/next_version/m3/audit_partial.py')
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    audit = audit_module.audit(campaign)
    require(audit['verified_records'] == audit['expected_records'] == 15000 and
            audit['queue_status_counts'] == {'done': 25}, 'parent completeness failed')
    c.atomic(output / 'parent_integrity.json', audit)
    source = source_contract(manifest)
    import pyarrow.parquet as pq
    import nltk
    from scripts.eval.mab2_mab_bridge import _load_mab_modules
    parquet = next(Path(s) for s in manifest['input_sha256'] if s.endswith('.parquet'))
    benchmark = next(Path(s).parent.parent for s in manifest['input_sha256'] if s.endswith('/utils/templates.py'))
    external_chunker = benchmark / 'utils/eval_other_utils.py'
    external_sha = c.sha(external_chunker)
    chunker, _, _ = _load_mab_modules(benchmark)
    rows = [r for r in pq.read_table(parquet).to_pylist() if r['metadata']['source'] == 'eventqa_65536']
    require(len(rows) == 5, 'wrong number of source contexts')
    chunk_records, evidence, schemas = [], [], []
    for ctx, row in enumerate(rows):
        identity = c.read(campaign / 'jobs' / f's42-c{ctx}' / 'context_identity.json')
        require('eventqa-' + text_sha(row['context'])[:16] == identity['context_id'], 'context source drift')
        chunks = chunker(row['context'], chunk_size=manifest['protocol']['chunk_size'])
        verify_chunks(chunks, identity['chunk_sha256'])
        spans = chunk_spans(row['context'], chunks, nltk.sent_tokenize(row['context']))
        chunk_records.extend(dict(context_index=ctx, context_id=identity['context_id'],
                                  chunk_sha256=identity['chunk_sha256'][i], **s)
                             for i, s in enumerate(spans))
        metadata = row['metadata']
        schemas.append(dict(context_index=ctx, top_level_fields=sorted(row),
                            metadata_fields={k: ('null' if v is None else type(v).__name__) for k, v in metadata.items()},
                            verified_source_span_fields=[],
                            qa_pair_ids_are_question_ids_not_source_offsets=True))
        require(len(row['questions']) == len(row['answers']) == 100, 'wrong question denominator')
        for q, (question, golds) in enumerate(zip(row['questions'], row['answers'])):
            require(text_sha(question) == identity['question_identity'][q]['question_text_hash'], 'question identity drift')
            saved = c.read(campaign / 'jobs' / f's42-c{ctx}' / 'queries' / f'q{q:03d}_full.json')['record']
            require(saved['gold_answers'] == golds, 'gold drift from original dataset')
            evidence.append(dict(context_index=ctx, query_id=q, context_id=identity['context_id'],
                                 question=question, gold_answers=golds,
                                 qa_pair_id=(metadata.get('qa_pair_ids') or [None] * 100)[q],
                                 question_text_sha256=text_sha(question),
                                 **locate_gold(row['context'], golds, spans)))
    provenance, lifecycle, associations, actions = [], {}, [], Counter()
    for job in manifest['jobs']:
        directory = campaign / 'jobs' / job['id']
        identity = c.read(directory / 'context_identity.json')
        construction = c.read(directory / 'construction.json')
        hashes = identity['chunk_sha256']
        reference = [r['chunk_sha256'] for r in chunk_records if r['context_index'] == job['context_index']]
        require(hashes == reference, 'source chunks differ across seeds')
        slots, events = replay(construction['construction_turn_diagnostics'], construction['pre_query_bank_summary'], hashes)
        expected_last_write = [s['last_write_step'] for s in slots]
        for path in (directory / 'queries').glob('q*.json'):
            trace = c.read(path)['record']['selection'][0]
            require(trace['last_write'] == expected_last_write, 'last-write replay mismatch with query trace')
        for slot in slots:
            provenance.append(dict(job_id=job['id'], seed=job['seed'], context_index=job['context_index'], **slot))
        lifecycle[job['id']] = dict(status='PASS', slot_count=len(slots), construction_turns=len(events), events=events)
        actions.update(t['action'] for t in events)
        for row in evidence:
            if row['context_index'] == job['context_index']:
                associations.append(dict(job_id=job['id'], seed=job['seed'], context_index=row['context_index'],
                                         query_id=row['query_id'], source_evidence_overlap=None,
                                         latent_semantic_relevance=None,
                                         unknown_reason=row['unknown_reason'], denominator_unit='seed/question'))
    require(len(evidence) == 500 and len(associations) == 2500, 'independent/repeated denominator mismatch')
    require(len({(r['context_index'], r['query_id']) for r in evidence}) == 500,
            'duplicate independent question identity')
    require('torch' not in sys.modules, 'unexpected Torch import')
    require(time.monotonic() - start < 1800, '30-minute audit budget exceeded')
    require(c.sha(external_chunker) == external_sha, 'chunker source changed during audit')
    check_unchanged(before, campaign)
    jsonl(output / 'provenance.jsonl', provenance)
    jsonl(output / 'evidence_mapping.jsonl', evidence)
    jsonl(output / 'seed_question_associations.jsonl', associations)
    jsonl(output / 'chunk_offsets.jsonl', chunk_records)
    jsonl(output / 'blind_sample.jsonl', blind_sample(evidence))
    c.atomic(output / 'lifecycle.json', lifecycle)
    c.atomic(output / 'schema_audit.json', schemas)
    coverage = dict(status='COMPLETE_WITH_LIMITATIONS', rubric=RUBRIC, jobs_verified=25,
                    parent_records_verified=15000, live_slot_versions=len(provenance),
                    live_slots_with_source_chunk=len(provenance), construction_write_actions=dict(actions),
                    unique_contexts=5, unique_questions=500, seed_question_associations=2500,
                    source_evidence_resolved_questions=0, unknown_source_evidence_questions=500,
                    unknown_seed_question_associations=2500,
                    unknown_latent_semantic_relevance_associations=2500,
                    lexical_status_counts=dict(Counter(r['lexical_status'] for r in evidence)),
                    lexical_cross_boundary_matches=sum(m['crosses_chunk_boundary'] for r in evidence for m in r['lexical_matches']),
                    parent_preserved=True, parent_files_checked=len(before), torch_loaded=False,
                    elapsed_seconds=time.monotonic() - start, manual_annotations_performed=0,
                    next_route='plan M4-B trace/output diagnostics; source semantic labeling requires a separate verified-span protocol',
                    manuscript_takeaway='Slot input provenance is recoverable; question evidence and latent retention are not established by these logs.')
    c.atomic(output / 'coverage.json', coverage)
    input_files = [p for p in campaign.rglob('*.json') if p.is_file()]
    analysis_manifest = dict(schema='m4-audit/v1', created_at=c.now(), parent_campaign=str(campaign),
                             parent_contract_sha256=CONTRACT, rubric=RUBRIC,
                             source_contract=source, analyzer_sha256=c.sha(__file__),
                             parent_json_sha256={str(p.relative_to(campaign)): c.sha(p) for p in input_files},
                             external_chunker_sha256=external_sha, python=sys.executable,
                             python_version=platform.python_version(),
                             paper_role='supporting', main_or_appendix='appendix', section_id='retrieval_diagnostics',
                             exp_id='M4-A', item_id='M4-A', claim_links=['R03', 'R04', 'R07'],
                             selected_outline_ref='research_notes/next_version/EXPERIMENT_PLAN.md',
                             research_questions=['RQ-M4'], experimental_designs=['offline provenance audit'],
                             todo_items=['M4-A1', 'M4-A2', 'M4-A3', 'M4-A4'],
                             fixed_conditions='M3 bank, model, questions, generation, policies unchanged',
                             comparability='offline inspection only; no new performance comparison',
                             source_label_rule='null for unverified spans; exact gold matches lexical only',
                             sample_rule='sha256(M4-A-20261008|context|query), ten per context; unannotated blind bundle')
    check_unchanged(before, campaign)
    c.atomic(output / 'manifest.json', analysis_manifest)
    c.atomic(output / 'status.json', dict(status=coverage['status'], finished_at=c.now()))
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    existed_before = destination.exists()
    try:
        run(Path(args.campaign).resolve(), destination)
    except Exception as exc:
        # Never mutate an existing successful audit or anything in the parent.
        if not existed_before and destination.exists() and (destination / 'status.json').exists():
            state = c.read(destination / 'status.json')
            if state.get('status') == 'RUNNING':
                c.atomic(destination / 'status.json', dict(status='FAILED', error=f'{type(exc).__name__}: {exc}'))
        raise
