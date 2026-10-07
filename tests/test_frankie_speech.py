from types import SimpleNamespace
import unittest
import numpy as np
from mtplx.frankie.speech import phrase_complete
from mtplx.frankie.breeze import BreezeMouth


class Boundaries(unittest.TestCase):
    def test_fast_first_phrase(self):
        self.assertTrue(phrase_complete('Great question!', ' So', first=True))
        self.assertTrue(phrase_complete('The wire hums a low,', ' electric', first=True))

    def test_continuation_keeps_short_sentences_and_clauses_together(self):
        self.assertFalse(phrase_complete('It twitches.', ' The', first=False))
        self.assertFalse(phrase_complete('The wire hums a low,', ' electric', first=False))
        self.assertTrue(phrase_complete('It twitches. The wire hums a low, electric tune.', ' Below', first=False))

    def test_initial_and_closing_quote(self):
        self.assertFalse(phrase_complete('The Love Song of J.', ' Alfred', first=True))
        self.assertTrue(phrase_complete('That was the last thing she said."', ' Then', first=False))

    def test_control_tokens_and_word_limits(self):
        self.assertTrue(phrase_complete('I will check', '<tool_call>', first=True))
        self.assertTrue(phrase_complete('I will check', '<think>', first=False))
        self.assertFalse(phrase_complete(' '.join(['word']*16), ' next', first=False))
        self.assertTrue(phrase_complete(' '.join(['word']*32), ' next', first=False))
        self.assertTrue(phrase_complete(' '.join(['word']*16), ' next', first=True))
        self.assertFalse(phrase_complete('Hello.', 'ing', first=True))


class EmptySpeech(unittest.TestCase):
    def mouth(self, attempts):
        class Model:
            calls=0; resets=0; decoder_resets=0; closed=0
            def __init__(self):
                self.audio_tokenizer=SimpleNamespace(decoder=SimpleNamespace(reset_streaming_state=self.reset_decoder))
            def _text_ids(self,text):return [1,2,3]
            def reset_speech_context(self):self.resets+=1
            def reset_decoder(self):self.decoder_resets+=1
            def generate(self,*a,**kw):
                sequence=attempts[self.calls];self.calls+=1
                try:
                    for count,pcm in sequence:yield SimpleNamespace(token_count=count,audio=np.array(pcm,dtype=np.float32))
                finally:self.closed+=1
        mouth = BreezeMouth.__new__(BreezeMouth)
        mouth.model = Model()
        mouth.codec_context = None
        mouth.instruction = lambda states: 'Speak naturally.'
        mouth.reference_db = -10
        return mouth

    def test_retry_before_any_pcm(self):
        m=self.mouth([[(0,[0,0])],[(1,[.2,.3])]])
        pcm=list(m.speak('Hello',[]))
        self.assertEqual(len(pcm),1);np.testing.assert_allclose(pcm[0],[.2,.3])
        self.assertEqual((m.model.calls,m.model.resets,m.model.closed),(2,1,2))

    def test_normal_path_unchanged(self):
        m=self.mouth([[(1,[.2,.3]),(1,[.3,.2])]])
        self.assertEqual(len(list(m.speak('Hello',[]))),2)
        self.assertEqual((m.model.calls,m.model.resets),(1,0))

    def test_second_empty_is_visible_failure(self):
        m=self.mouth([[(0,[0])],[(0,[0])]])
        with self.assertRaisesRegex(RuntimeError,'no speech'):
            list(m.speak('Hello',[]))
        self.assertEqual((m.model.calls,m.model.resets),(2,2))

    def test_no_retry_after_published_audio(self):
        m=self.mouth([[(1,[.2]),(0,[0])]])
        g=m.speak('Hello',[]);next(g)
        with self.assertRaisesRegex(RuntimeError,'invalid empty'):next(g)
        self.assertEqual(m.model.calls,1)

    def test_cancellation_closes_generator_and_decoder(self):
        m=self.mouth([[(1,[.2]),(1,[.3])]])
        g=m.speak('Hello',[]);next(g);g.close()
        self.assertEqual((m.model.closed,m.model.decoder_resets),(1,1))

    def test_nonfinite_is_not_retried(self):
        m=self.mouth([[(1,[float('nan')])]])
        with self.assertRaisesRegex(RuntimeError,'non-finite'):
            list(m.speak('Hello',[]))
        self.assertEqual(m.model.calls,1)


if __name__=='__main__':unittest.main()
