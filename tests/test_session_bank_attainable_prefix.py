"""Near-prefix restore under boundary-true restoration: a tiny-gap recurrent candidate is taken only when its
attainable checkpoint beats the exact-prefix floor. Executes the native function ASTs with CPU-only storage doubles,
never a model."""
import ast
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import unittest
from unittest.mock import patch

MTPLX = Path(__file__).resolve().parents[1] / 'mtplx'


def load_functions(path, requests, namespace):
    tree = ast.parse(path.read_text())
    nodes = []
    for parent, name in requests:
        scope = tree.body if parent is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent)
        node = next(n for n in scope if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        nodes.append(node)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


def native_namespace():
    ns = dict(os=os, sys=sys, time=time, inspect=inspect, DEFAULT_PREFIX_BLOCK_SIZE=256, DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS=512,
              block_prefix_restore_enabled=lambda: True,
              CacheMissReason=SimpleNamespace(PREFIX_DIVERGENCE_AT_TOKEN=SimpleNamespace(value='divergence'), NEW_SESSION=SimpleNamespace(value='new')))
    bank_functions = ['_boundary_true_restore_enabled', '_policy_uses_committed_history', '_mtp_history_policy_compatible', '_restore_identity_compatible', 'common_prefix_len', 'block_aligned_prefix_len']
    load_functions(MTPLX/'session_bank.py', [(None, n) for n in bank_functions] + [('SessionBankEntry', '_ensure_boundaries_loaded'), ('SessionBankEntry', 'recurrent_boundary_at_or_below'), ('SessionBank', 'near_prefix_candidates')], ns)
    gen_functions = ['_env_int', '_env_falsey', '_normalize_mtp_history_policy', '_mtp_history_uses_committed_cache', '_near_prefix_restore_enabled', '_entry_matches_restore_lookup', '_accepts_served_out', 'PostcommitAbort', '_check_postcommit_abort', '_restore_near_prefix_prompt_state']
    load_functions(MTPLX/'generation.py', [(None, n) for n in gen_functions], ns)
    return ns


def entry(ns, *, recurrent=True, boundary=559, length=1241, matched=1235, **updates):
    # 1,258 actual request IDs; terminal six entry tokens differ.
    tokens = tuple(range(matched)) + tuple(-i-1 for i in range(length-matched))
    e = SimpleNamespace(token_ids=tokens, prefix_len=length, has_recurrent=recurrent,
                        gdn_boundaries=[] if boundary is None else [(boundary, object(), object())],
                        gdn_boundary_loader=None, model_path='model', mtp_enabled=True,
                        hidden_variant='native', template_hash='template', mtp_history_policy='committed',
                        draft_head_identity='draft', policy_fingerprint='policy',
                        mtp_history_snapshot=object(), mtp_history_cache_ref=None,
                        mtp_snapshot_epoch=3, snapshot_epoch=3, live_ref_only=False,
                        cache_ref=None, token_hash='near-entry', session_id='s', cache_source='ram')
    e._ensure_boundaries_loaded = lambda: ns['_ensure_boundaries_loaded'](e)
    e.recurrent_boundary_at_or_below = lambda n: ns['recurrent_boundary_at_or_below'](e, n)
    e.__dict__.update(updates)
    return e


class Bank:
    SUPPORTS_NEAR_PREFIX_MIN_RESTORE = True
    def __init__(self, entries):
        self.entries = entries
        self._entries = {e.token_hash: e for e in entries}
        self.calls = []
        self.cold_queries = []
    def near_prefix_candidates(self, tokens, **kwargs):
        self.query = kwargs
        return [(e, next((i for i, (a, b) in enumerate(zip(tokens, e.token_ids)) if a != b), min(len(tokens), len(e.token_ids)))) for e in self.entries]
    def restore_entry_prefix_cache(self, rt, e, matched, *, mode, cache_factory, served_out):
        # Record a real restore attempt at the native call boundary. Returning
        # None prevents tensor/model work and exercises native fallback modes.
        self.calls.append((e, matched, mode, cache_factory, served_out.copy()))
        return None
    def _purge_expired(self):
        pass
    def _cold_near_prefix_candidate(self, tokens, **kwargs):
        self.cold_queries.append(kwargs)
        return None


def generation_attempt(ns, entries, *, floor=1157, ceiling=None, abort=None):
    bank = Bank(entries)
    fn = ns['_restore_near_prefix_prompt_state']
    gen = fn(SimpleNamespace(model_path='model', mtp_enabled=True), list(range(1258)),
             base_hidden_variant='native', mtp_hidden_variant='native', mtp_history_policy='committed',
             session_bank=bank, template_hash='template', draft_head_identity='draft', policy_fingerprint='policy',
             min_restore_tokens=floor, matched_ceiling=ceiling, abort_check=abort)
    if list(gen):
        raise AssertionError('Unexpected model or progress work')
    return bank


def shadow_attempt(ns, entries, floor=1157):
    bank = Bank(entries)
    result = ns['near_prefix_candidates'](bank, list(range(1258)), max_token_gap=8,
        min_matched_tokens=64, block_size=256, block_min_matched_tokens=512,
        allow_block_prefix=True, model_path='model', mtp_enabled=True,
        hidden_variant='native', template_hash='template', mtp_history_policy='committed',
        draft_head_identity='draft', policy_fingerprint='policy', min_restore_tokens=floor)
    return bank, result


class AttainableTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'MTPLX_SESSION_BOUNDARY_TRUE_RESTORE': '1', 'MTPLX_SESSION_NEAR_PREFIX_MAX_TOKEN_GAP': '8', 'MTPLX_SESSION_NEAR_PREFIX_RESTORE': '1'})
        self.env.start(); self.addCleanup(self.env.stop)
        self.ns = native_namespace()

    def test_tiny_gap_bad_boundary_rejected_before_restore_or_shadow(self):
        e = entry(self.ns, cache_ref=object())
        self.assertEqual(generation_attempt(self.ns, [e]).calls, [])
        bank, result = shadow_attempt(self.ns, [e])
        self.assertEqual(result, [(e, 1235)])  # Enumeration/identity is unchanged.
        self.assertIsNone(bank.cold_queries[0]['resident_duplicates'])
        self.assertEqual(bank.cold_queries[0]['min_useful_matched_tokens'], 1158)

    def test_newest_boundary_better_than_floor_accepts_original_modes(self):
        e = entry(self.ns, boundary=1200, cache_ref=object())
        calls = generation_attempt(self.ns, [e]).calls
        self.assertEqual([c[2] for c in calls], ['reference', 'clone'])
        self.assertTrue(all(c[0] is e and c[1] == 1235 and c[3] is None and c[4] == {} for c in calls))
        bank, _ = shadow_attempt(self.ns, [e])
        self.assertIn(e.token_hash, bank.cold_queries[0]['resident_duplicates'])
        self.assertEqual(bank.cold_queries[0]['min_useful_matched_tokens'], 1235)

    def test_equal_missing_zero_floor_and_lazy_boundary(self):
        for boundary, floor in [(1157, 1157), (None, 1157), (None, 0)]:
            with self.subTest(boundary=boundary, floor=floor):
                e = entry(self.ns, boundary=boundary)
                self.assertEqual(generation_attempt(self.ns, [e], floor=floor).calls, [])
                b, _ = shadow_attempt(self.ns, [e], floor=floor)
                self.assertIsNone(b.cold_queries[0]['resident_duplicates'])
        e = entry(self.ns, boundary=None); calls = []
        e.gdn_boundary_loader = lambda: calls.append('load') or [(1200, object(), object())]
        self.assertEqual(len(generation_attempt(self.ns, [e]).calls), 1)
        self.assertEqual(calls, ['load'])

    def test_pure_attention_and_explicit_legacy_off_switch_unchanged(self):
        for recurrent, off in [(False, False), (True, True)]:
            with self.subTest(recurrent=recurrent, off=off), patch.dict(os.environ, {'MTPLX_SESSION_BOUNDARY_TRUE_RESTORE': '0' if off else '1'}):
                e = entry(self.ns, recurrent=recurrent, boundary=None)
                self.assertEqual(len(generation_attempt(self.ns, [e]).calls), 1)
                b, _ = shadow_attempt(self.ns, [e])
                self.assertIn(e.token_hash, b.cold_queries[0]['resident_duplicates'])

    def test_large_gap_and_identity_mtp_guards_preserved(self):
        for off in ('0', '1'):
            with patch.dict(os.environ, {'MTPLX_SESSION_BOUNDARY_TRUE_RESTORE': off}):
                e = entry(self.ns, matched=1200)
                self.assertEqual(generation_attempt(self.ns, [e]).calls, [])
                b, _ = shadow_attempt(self.ns, [e]); self.assertIsNone(b.cold_queries[0]['resident_duplicates'])
        for changes in ({'model_path': 'wrong'}, {'mtp_history_snapshot': None}, {'mtp_snapshot_epoch': 4}):
            with self.subTest(changes=changes):
                e = entry(self.ns, boundary=1200, **changes)
                self.assertEqual(generation_attempt(self.ns, [e]).calls, [])
                b, _ = shadow_attempt(self.ns, [e]); self.assertIsNone(b.cold_queries[0]['resident_duplicates'])

    def test_skip_bad_then_try_better_preserves_media_ceiling_and_abort(self):
        bad, good = entry(self.ns), entry(self.ns, boundary=1200, token_hash='good')
        self.assertEqual([c[0] for c in generation_attempt(self.ns, [bad, good]).calls], [good])
        self.assertEqual(generation_attempt(self.ns, [good], ceiling=1190).calls, [])
        with self.assertRaisesRegex(self.ns['PostcommitAbort'], 'foreground_preempted_postcommit'):
            generation_attempt(self.ns, [good], abort=lambda: True)

    def test_bad_resident_does_not_hide_cold_alternative(self):
        e = entry(self.ns); cold = entry(self.ns, boundary=1200, token_hash='cold')
        bank = Bank([e]); observed = []
        def cold_lookup(tokens, **kwargs):
            observed.append(kwargs)
            # Cold backend owns the actual hydration; this double proves the
            # changed caller passes neither an unusable twin nor an inflated floor.
            self.assertIsNone(kwargs['resident_duplicates'])
            self.assertEqual(kwargs['min_useful_matched_tokens'], 1158)
            return cold, 1210
        bank._cold_near_prefix_candidate = cold_lookup
        result = self.ns['near_prefix_candidates'](bank, list(range(1258)), max_token_gap=8,
            min_matched_tokens=64, allow_block_prefix=True, model_path='model', mtp_enabled=True,
            hidden_variant='native', template_hash='template', mtp_history_policy='committed',
            draft_head_identity='draft', policy_fingerprint='policy', min_restore_tokens=1157)
        self.assertEqual(len(observed), 1)
        self.assertIn((cold, 1210), result)


if __name__ == '__main__':
    unittest.main()
