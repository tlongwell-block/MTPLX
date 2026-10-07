"""Idle history warming: exercise Session's actual cache methods without loading any models."""
import ast
import asyncio
from concurrent.futures import Future
import copy
import logging
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'mtplx/frankie/session.py'
tree = ast.parse(SOURCE.read_text())
original_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Session')
names = {'spawn', 'invalidate_history_prefill', 'prepare_idle_history', 'rollback_unheard',
         'yield_prefix', 'generate'}
methods = [n for n in original_class.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
           and n.name in names]
namespace = dict(asyncio=asyncio, copy=copy, logging=logging, Event=Event,
                 capture_draft=lambda run: {'text': 'Unheard private draft.'})
module = ast.Module(body=[*[n for n in tree.body if isinstance(n, ast.FunctionDef)
                           and n.name in {'snapshot_history', 'public'}],
                         ast.ClassDef(name='Session', bases=[], keywords=[], body=methods,
                                      decorator_list=[])], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
Session = namespace['Session']


class Scheduler:
    def __init__(self):
        self.jobs = []
        self.foreground = 0

    def submit_idle_postcommit(self, fn):
        future = Future()
        self.jobs.append((fn, future))
        return future

    def foreground_pending(self):
        return self.foreground

    def execute(self):
        fn, future = self.jobs.pop(0)
        try:
            future.set_result(fn())
        except BaseException as error:
            future.set_exception(error)


class CacheTests(unittest.IsolatedAsyncioTestCase):
    def session(self, *, interrupted=True, done=True):
        s = Session()
        s.id = 'session'
        s.executor = Scheduler()
        s.calls = []

        def warm(settings, **kwargs):
            if kwargs['abort_check']():
                return None
            self.assertEqual(kwargs['prefill_step_size'](), 32)
            s.calls.append(kwargs['history'])
            return object()

        s.engine = SimpleNamespace(brain_interface='text', warm=warm)
        item = {'type': 'message', 'role': 'assistant', 'id': 'assistant',
                'status': 'incomplete' if interrupted else 'completed',
                'content': [{'type': 'output_audio', 'transcript': 'heard ' * 30}]}
        abort = Event()
        if interrupted:
            abort.set()
            item['_playback_interrupted'] = True
            item['_interrupted_draft'] = {'text': 'Unheard private draft.'}
        s.current = SimpleNamespace(id='response', done=done, abort=abort,
                                    interrupted=interrupted, item=item, tool_items={},
                                    playback_finished=not interrupted)
        s.items = [item]
        s.tasks = set()
        s.history_prefill_enabled = True
        s.history_prefill_revision = 0
        s.history_prefill_key = None
        s.history_prefill_task = None
        s.history_prefill_abort = Event()
        s.closed = False
        s.listening = interrupted
        s.manual_size = 0
        s.semantic_pending = False
        s.queued_task_response = False
        s.user_revision = s.response_revision = 2
        s.settings = {'background_tasks': False}
        s.listener = None
        return s

    async def settle(self):
        for _ in range(4):
            await asyncio.sleep(0)

    async def finish(self, s):
        await self.settle()
        s.executor.execute()
        await self.settle()

    async def test_interrupted_history_warms_while_user_speaks(self):
        s = self.session()
        s.prepare_idle_history()
        await self.finish(s)
        self.assertEqual(len(s.calls), 1)
        history = s.calls[0]
        self.assertNotIn('Unheard', history[0]['content'][0]['transcript'])
        self.assertEqual(history[0]['_interrupted_draft']['text'], 'Unheard private draft.')
        self.assertIsNotNone(s.history_prefill_key)

    async def test_completed_response_needs_playback_drain(self):
        s = self.session(interrupted=False)
        s.current.playback_finished = False
        s.prepare_idle_history()
        self.assertIsNone(s.history_prefill_task)
        s.current.playback_finished = True
        s.prepare_idle_history()
        await self.finish(s)
        self.assertEqual(len(s.calls), 1)

    async def test_cancel_waits_for_generation_settlement(self):
        s = self.session(done=False)
        s.prepare_idle_history()
        self.assertIsNone(s.history_prefill_task)
        s.current.done = True
        s.prepare_idle_history()
        await self.finish(s)
        self.assertEqual(len(s.calls), 1)

    async def test_one_pending_job_and_success_deduplication(self):
        s = self.session()
        s.prepare_idle_history()
        s.prepare_idle_history()
        await self.settle()
        self.assertEqual(len(s.executor.jobs), 1)
        s.executor.execute()
        await self.settle()
        s.prepare_idle_history()
        await self.settle()
        self.assertEqual(s.executor.jobs, [])

    async def test_late_truncate_replaces_queued_snapshot(self):
        s = self.session()
        s.prepare_idle_history()
        await self.settle()
        s.invalidate_history_prefill()
        s.items[0]['content'][0]['transcript'] = 'New acknowledged frontier.'
        s.prepare_idle_history()
        s.executor.execute()
        await self.settle()
        self.assertEqual(s.calls, [])
        self.assertIsNone(s.history_prefill_key)
        self.assertEqual(len(s.executor.jobs), 1)
        s.executor.execute()
        await self.settle()
        self.assertEqual(s.calls[0][0]['content'][0]['transcript'], 'New acknowledged frontier.')

    async def test_foreground_wins_and_does_not_mark_warm_or_spin(self):
        s = self.session()
        s.prepare_idle_history()
        s.executor.foreground = 1
        await self.finish(s)
        self.assertEqual(s.calls, [])
        self.assertIsNone(s.history_prefill_key)
        self.assertEqual(s.executor.jobs, [])
        s.executor.foreground = 0
        s.prepare_idle_history()
        await self.finish(s)
        self.assertEqual(len(s.calls), 1)

    async def test_stale_worker_completion_cannot_mark_revision_warm(self):
        s = self.session()
        calls = []
        def warm(*args, **kwargs):
            calls.append(1)
            s.invalidate_history_prefill()
            s.closed = True
            return object()
        s.engine.warm = warm
        s.prepare_idle_history()
        await self.finish(s)
        self.assertEqual(calls, [1])
        self.assertIsNone(s.history_prefill_key)
        self.assertEqual(s.executor.jobs, [])

    async def test_new_response_or_input_revision_invalidates_work(self):
        for mutate in (lambda s: setattr(s, 'current', None),
                       lambda s: setattr(s, 'user_revision', 3)):
            s = self.session()
            s.prepare_idle_history()
            mutate(s)
            await self.finish(s)
            self.assertEqual(s.calls, [])

    async def test_never_triggers_new_asr_or_vision(self):
        for part in ({'type': 'input_audio'}, {'type': 'input_image'}):
            s = self.session()
            s.items.insert(0, {'role': 'user', 'content': [part]})
            s.prepare_idle_history()
            self.assertIsNone(s.history_prefill_task)
        s = self.session()
        s.items.insert(0, {'role': 'user', 'content': [{'type': 'input_audio', '_transcript': 'ready'}]})
        s.prepare_idle_history()
        await self.finish(s)
        self.assertEqual(len(s.calls), 1)

    async def test_no_idle_scheduler_or_pending_tools_skips_work(self):
        s = self.session()
        s.executor = object()
        s.prepare_idle_history()
        self.assertIsNone(s.history_prefill_task)
        s = self.session()
        s.current.tool_items['call'] = {}
        s.prepare_idle_history()
        self.assertIsNone(s.history_prefill_task)

    async def test_normal_small_reply_stays_unchanged(self):
        s = self.session(interrupted=False)
        s.items[0]['content'][0]['transcript'] = 'Okay.'
        s.prepare_idle_history()
        self.assertIsNone(s.history_prefill_task)

    async def test_actual_rollback_only_warms_acknowledged_speech(self):
        s = self.session(interrupted=False)
        run = s.current
        run.ready = Event()
        run.playback_paused = True
        run.played_ms = 100
        run.chunks = [{'text': 'Heard.', 'end_ms': 90}, {'text': 'Unheard.', 'end_ms': 200}]
        s.playback_wake = None
        s.metrics = {'barge_ins': 0}
        s.event = lambda *args, **kwargs: None
        s.settle_task_results = lambda run: None
        s.rollback_unheard(run)
        await self.finish(s)
        self.assertEqual(s.calls[0][0]['content'][0]['transcript'], 'Heard.')
        self.assertTrue(run.abort.is_set())

    async def test_real_prefix_yield_triggers_warm_after_done_response(self):
        s = self.session(interrupted=False)
        run = s.current
        run.ready = Event()
        run.playback_finished = False
        run.playback_paused = False
        run.played_ms = 100
        run.chunks = [{'text': 'Confirmed heard.', 'end_ms': 90}]
        s.listening = True
        s.overlap_run = run
        s.settings['turn_detection'] = {'interrupt_response': True}
        s.playback_wake = None
        s.metrics = {'barge_ins': 0}
        s.event = lambda *args, **kwargs: None
        s.settle_task_results = lambda run: None
        s.cancel_prefix = lambda: None
        self.assertTrue(s.yield_prefix(run, 'User took the floor.'))
        await self.finish(s)
        self.assertEqual(s.calls[0][0]['content'][0]['transcript'], 'Confirmed heard.')
        self.assertIsNone(s.overlap_run)

    async def test_real_cancelled_generation_settles_before_warm(self):
        s = self.session(done=False)
        run = s.current
        run.input_pending = False
        run.ready = Event()
        run.ready.set()
        run.visible = True
        run.tool_error = None
        run.item_id = 'assistant'
        s.settings['output_modalities'] = ['audio']
        run.settings = copy.deepcopy(s.settings)
        s.events = []
        s.event = lambda *args, **kwargs: s.events.append((args, kwargs))
        s.settle_task_results = lambda run: None
        s.maybe_start_task_response = lambda: None
        foreground = []
        def submit(fn, *args):
            future = Future()
            foreground.append((fn, args, future))
            return future
        s.executor.submit = submit
        s.loop = asyncio.get_running_loop()
        task = asyncio.create_task(s.generate(run, s.items))
        await self.settle()
        fn, args, future = foreground.pop()
        future.set_result(fn(*args))
        await self.settle()
        await task
        self.assertTrue(run.done)
        self.assertEqual(run.item['status'], 'incomplete')
        self.assertEqual(s.events[-1][0], ('response.done',))
        self.assertEqual(len(s.executor.jobs), 1)
        s.executor.execute()
        await self.settle()
        self.assertEqual(len(s.calls), 1)


if __name__ == '__main__':
    unittest.main()
