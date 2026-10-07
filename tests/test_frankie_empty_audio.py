import asyncio
from threading import Event
import numpy as np
import pytest
from test_frankie_session import setup


def events(s):
    return list(s.outgoing._queue)


def engine(s, text='', slow=False, failure=False):
    s.engine.brain_interface='text'
    entered, release=Event(),Event()
    if not slow:release.set()
    calls=[]
    original=s.engine.respond
    def prepare(item):
        entered.set()
        assert release.wait(2), 'test preparation blocked'
        if failure:raise ValueError('transcription failed')
        parts=[]
        for i,p in enumerate(item['content']):
            if p['type']=='input_audio':
                p.setdefault('_transcript',text)
                parts.append((item,i,p['_transcript']))
        return parts
    def respond(*args,**kw):
        calls.append(args)
        return original(*args,**kw)
    s.engine.prepare_audio=prepare;s.engine.respond=respond
    return entered,release,calls


@pytest.mark.parametrize('release_before_show',[True,False])
@pytest.mark.parametrize('text',['','  \n','Hello.'])
def test_empty_automatic_turn_never_exposed_and_real_speech_still_works(text,release_before_show):
    async def check(s):
        entered,release,calls=engine(s,text,slow=True)
        run=s.start(s.audio_item(np.zeros(2400)),tentative=True)
        assert await asyncio.to_thread(entered.wait,1)
        if release_before_show:
            release.set();await asyncio.sleep(.02)
        s.show(run);s.spec=None
        if not release_before_show:
            assert not run.visible
            release.set()
        await asyncio.wait_for(asyncio.gather(*s.tasks),2)
        types=[e['type'] for e in events(s)]
        if text.strip():
            assert len(calls)==1 and run.visible and run.status=='completed'
            assert types.count('response.created')==types.count('response.done')==1
            assert run.text=='Hello.'
        else:
            assert not calls and not run.visible and not s.items and run.done
            assert 'response.created' not in types
            assert types.count('frankie.input.ignored')==1
            assert s.user_revision==0
    asyncio.run(setup(check))


def test_resume_during_transcription_does_not_publish_old_response():
    async def check(s):
        entered,release,calls=engine(s,'Hello.',slow=True)
        run=s.start(s.audio_item(np.zeros(2400)),tentative=True)
        assert await asyncio.to_thread(entered.wait,1)
        s.show(run);s.discard_spec();release.set()
        await asyncio.wait_for(asyncio.gather(*s.tasks),2)
        assert not run.visible and not s.items
        assert 'response.created' not in [e['type'] for e in events(s)]
    asyncio.run(setup(check))


def test_transcription_failure_settles_after_commit():
    async def check(s):
        entered,release,calls=engine(s,slow=True,failure=True)
        run=s.start(s.audio_item(np.zeros(2400)),tentative=True)
        assert await asyncio.to_thread(entered.wait,1)
        s.show(run);s.spec=None;release.set()
        await asyncio.wait_for(asyncio.gather(*s.tasks),2)
        assert run.status=='failed'
        assert not calls
        assert [e['type'] for e in events(s)].count('response.done')==1
    asyncio.run(setup(check))


def test_neural_audio_does_not_require_a_transcript():
    async def check(s):
        _,_,calls=engine(s,'')
        s.engine.brain_interface='neural'
        run=s.start(s.audio_item(np.zeros(2400)),tentative=True)
        s.show(run);s.spec=None
        await asyncio.wait_for(asyncio.gather(*s.tasks),2)
        assert calls and run.status=='completed'
    asyncio.run(setup(check))


def test_explicit_request_is_not_silenced():
    async def check(s):
        _,_,calls=engine(s,'')
        run=s.start(s.audio_item(np.zeros(2400)))
        await asyncio.wait_for(asyncio.gather(*s.tasks),2)
        assert calls and run.status=='completed'
    asyncio.run(setup(check))


def test_repeated_empty_vad_turns_do_not_poison_next_real_turn():
    from test_frankie_streaming_session import feed
    from test_frankie_background_session import wait_for
    async def check(s):
        for _ in range(2):
            _,_,calls=engine(s,'')
            await feed(s,16000,6)
            await feed(s,0,12)
            await wait_for(lambda:s.current is not None and s.current.done)
            assert not s.items and not calls and s.spec is None
        _,_,calls=engine(s,'Hello.')
        await feed(s,16000,6)
        await feed(s,0,12)
        await wait_for(lambda:s.current.done)
        assert s.current.status=='completed' and s.current.text=='Hello.'
        assert [e['type'] for e in events(s)].count('response.created')==1
        assert s.user_revision==1
    asyncio.run(setup(check))
