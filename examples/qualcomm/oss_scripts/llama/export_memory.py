# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock


_compact_provenance = ContextVar("qwen35_compact_fx_provenance", default=False)
_provenance_lock = RLock()


@contextmanager
def compact_fx_provenance():
    """Keep direct FX node origins without copying recursive debug histories."""
    from torch.fx.traceback import NodeSource

    # PyTorch deep-copies the entire ancestry at each graph rewrite. Scope the
    # workaround to this export; other threads retain the original provenance.
    with _provenance_lock:
        original_init = NodeSource.__init__
        token = _compact_provenance.set(True)
        missing = object()

        def init_source(self, node, pass_name="", action=None):
            if not _compact_provenance.get():
                return original_init(self, node, pass_name, action)
            provenance = (
                node.meta.pop("from_node", missing) if node is not None else missing
            )
            try:
                original_init(self, node, pass_name, action)
            finally:
                if provenance is not missing:
                    node.meta["from_node"] = provenance

        try:
            NodeSource.__init__ = init_source
            yield
        finally:
            NodeSource.__init__ = original_init
            _compact_provenance.reset(token)
