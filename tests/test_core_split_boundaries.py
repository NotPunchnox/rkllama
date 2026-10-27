"""Keep core parser/lifecycle safety without experimental serving adapters."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path
import pytest
from rkllama.api.format_utils import get_tool_calls
from test_native_review_regressions import load_worker

TOOLS = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {
    'type': 'object', 'properties': {'city': {'type': 'string'}},
    'required': ['city'], 'additionalProperties': False}}}]

@pytest.mark.parametrize('text', [
    '{"name":"lookup","arguments":{"city":"A","city":"B"}}',
    '<tool_call>{"name":"lookup","arguments":{"city":"A","city":"B"}}</tool_call>',
])
def test_duplicate_tool_argument_keys_rejected_at_first_decode(text):
    assert get_tool_calls(text, TOOLS) == []

def test_abort_is_retried_until_native_thread_quiesces():
    worker = load_worker()
    model = SimpleNamespace(abort=Mock())
    state = {'alive': True, 'joins': 0}
    def join(timeout):
        state['joins'] += 1
        if state['joins'] == 3:
            state['alive'] = False
    thread = SimpleNamespace(is_alive=lambda: state['alive'], join=join)
    assert hasattr(worker, 'abort_and_join')
    assert worker.abort_and_join(model, thread, timeout=1)
    assert model.abort.call_count == 3

def test_hung_worker_does_not_wait_for_non_daemon_native_thread():
    worker = load_worker()
    assert hasattr(worker, 'terminate_unquiescent_worker')
    with patch.object(worker.os, '_exit', side_effect=SystemExit) as exit_process:
        with pytest.raises(SystemExit):
            worker.terminate_unquiescent_worker()
    exit_process.assert_called_once_with(1)

def test_prefix_cache_does_not_enable_output_validation_adapter():
    api = Path(__file__).parents[1] / 'src/rkllama/api'
    assert not (api / 'validated_completion.py').exists()
    assert (api / 'prefix_cache.py').exists()
