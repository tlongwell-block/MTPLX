"""The brain's hidden states, phrase by phrase, for brain-led delivery.

A request-scoped observer of states the brain has already computed: it wraps
the runtime's and the graph bank's dispatch boundaries (never a compiled
graph's body) and reuses CommittedFeatures' cache-boundary and token-id
acceptance rule. Nothing about generation, sampling, caches or parameters
changes. One observer owns a runtime at a time; other requests may share the
runtime, and are told apart by their cache identity. ``width`` is the width
of the states the brain's forward returns: 10240 for Flash (its widened
pre-mixer stream), 5120 for 27B (its post-norm final state).
"""
from functools import wraps
from mtplx.features import CommittedFeatures


class FinalFeatures(CommittedFeatures):
    def __init__(self, runtime, prompt_length, callback, *, width):
        super().__init__(runtime, prompt_length, callback)
        self.width = width
        self.runtime = runtime
        self.depth = 0
        self.restore = []
        self.observed_calls = 0
        self.foreign_calls = 0
        self.bound_cache = None

    def bind_state(self, state):
        """Only the owning generation passes this explicit request callback.

        Never infer ownership from offsets or token equality: concurrent text
        work can have both in common. A rebase changes the owner's cache but
        keeps matching already-executed rows; the new state's last row supplies
        an emitted token that had not yet executed before the rebase.
        """
        import mlx.core as mx
        prefix=state.token_prefix
        self.rows={p:row for p,row in self.rows.items() if p<len(prefix) and row[0]==prefix[p]}
        self.bound_cache=state.trunk_cache
        self.cache=state.trunk_cache
        if prefix and state.hidden is not None:
            self.record(mx.array([[int(prefix[-1])]]),state.hidden[:,-1:,:],self.cache,len(prefix)-1)

    def bind_graphbank(self, bank):
        """Observe only this generator's concrete dispatch object."""
        if bank.runtime is not self.runtime:
            raise RuntimeError('Graph bank belongs to a different runtime')
        if any(owner is bank for owner, *_ in self.restore):
            raise RuntimeError('Graph bank is already observed')
        self._wrap(bank, "forward_ar", default_hidden=True)
        self._wrap(bank, "forward_ar_capture", default_hidden=True)

    def _wrap(self, owner, name, *, default_hidden=False):
        original = getattr(owner, name)
        prior = owner.__dict__.get(name)
        had_own = name in owner.__dict__

        @wraps(original)
        def observed(*args, **kwargs):
            ids = args[0]
            cache = kwargs.get("cache")
            start = self.offset(cache)
            outer = self.depth == 0
            self.depth += 1
            try:
                result = original(*args, **kwargs)
            finally:
                self.depth -= 1
            if outer and cache is not None and cache is not self.bound_cache and self.bound_cache is not None:
                self.foreign_calls += 1
            if outer and cache is self.bound_cache and cache is not None and kwargs.get("return_hidden", default_hidden):
                if not isinstance(result, tuple) or len(result) < 2:
                    raise RuntimeError("Brain feature forward did not return hidden states")
                hidden = result[1]
                if hidden is not None:
                    if hidden.shape[:2] != ids.shape or hidden.shape[-1] != self.width:
                        raise RuntimeError(f"Unexpected brain feature shape: {hidden.shape}")
                    self.record(ids, hidden, cache, start)
                    self.observed_calls += 1
            return result

        setattr(owner, name, observed)
        self.restore.append((owner, name, had_own, prior))

    def __enter__(self):
        if getattr(self.runtime, "_expression_feature_owner", None) is not None:
            raise RuntimeError("Another feature observer owns this runtime")
        self.runtime._expression_feature_owner = self
        try:
            self._wrap(self.runtime, "forward_ar")
            self._wrap(self.runtime, "forward_ar_capture")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for owner, name, had_own, prior in reversed(self.restore):
            if had_own:
                setattr(owner, name, prior)
            else:
                delattr(owner, name)
        self.restore.clear()
        self.runtime._expression_feature_owner = None
        self.rows.clear()
        self.cache = None
        self.bound_cache = None
