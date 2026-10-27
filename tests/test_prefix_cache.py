"""Hardware-free cache state tests; native correctness is a separate gate."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest


def fake_model():
 from rkllama.api import callback
 model=SimpleNamespace(_prefix_save_path=None,rkllm_clear_kv_cache=Mock(return_value=0),handle=None,rkllm_load_prompt_cache=Mock(return_value=0),release_prompt_cache=Mock(return_value=0),calls=[])
 def run(mode,kind,inputs,options):
  ids,_=inputs;model.calls.append((list(ids),dict(options),model._prefix_save_path))
  if model._prefix_save_path:
   assert callback.suppress_output is True
   Path(model._prefix_save_path).write_bytes(b'fixture')
   callback.global_metrics[:]=[len(ids),1,10,1]
  else:
   assert not callback.global_text
   callback.global_text.append('ANSWER');callback.global_metrics[:]=[32,6,2,3]
  return 0
 model._run=run
 return model


def test_exact_prefix_is_reused_but_changed_prefix_is_rebuilt(tmp_path):
 from rkllama.api.prefix_cache import PrefixCache
 from rkllama.api import callback
 model=fake_model();cache=PrefixCache(model,tmp_path)
 ids=list(range(300));params=(0,1,(ids,None),{'max_new_tokens':16})
 callback.reset_callback_state();assert cache.invoke(*params)==0
 assert len(model.calls)==2 and model.calls[0][0]==ids[:-32] and model.calls[1][0]==ids
 assert cache.last_priming_metrics==[268,1,10,1]
 assert callback.global_text==['ANSWER'] and cache.last_hit is False
 callback.reset_callback_state();cache.invoke(*params)
 assert len(model.calls)==3 and cache.last_hit is True and cache.last_priming_metrics==[]
 changed=ids.copy();changed[0]=999
 callback.reset_callback_state();cache.invoke(0,1,(changed,None),{'max_new_tokens':16})
 assert len(model.calls)==5 and cache.last_hit is False
 root=cache.root.name;cache.close();assert not Path(root).exists()


def test_failed_priming_resets_suppression_and_does_not_run_full_query(tmp_path):
 from rkllama.api.prefix_cache import PrefixCache
 from rkllama.api import callback
 model=fake_model();calls=[]
 def failed(*args):calls.append(args);return -1
 model._run=failed;cache=PrefixCache(model,tmp_path)
 with pytest.raises(RuntimeError):cache.invoke(0,1,(list(range(300)),None),{'max_new_tokens':16})
 assert len(calls)==1 and callback.suppress_output is False and callback.global_text==[]
 cache.close()


def test_priming_callback_output_never_enters_http_buffers():
 from rkllama.api import callback
 from rkllama.api.classes import LLMCallState
 callback.reset_callback_state();callback.suppress_output=True
 try:
  callback.callback_impl(SimpleNamespace(contents=SimpleNamespace(text=b'PRIVATE PRIMING OUTPUT')),None,LLMCallState.RKLLM_RUN_NORMAL)
  assert callback.global_text==[]
 finally:callback.suppress_output=False;callback.reset_callback_state()


def test_short_request_releases_loaded_cache_and_does_not_reuse_it(tmp_path):
 from rkllama.api.prefix_cache import PrefixCache
 from rkllama.api import callback
 model=fake_model();cache=PrefixCache(model,tmp_path);ids=list(range(300))
 callback.reset_callback_state();cache.invoke(0,1,(ids,None),{})
 callback.reset_callback_state();cache.invoke(0,1,([1,2],None),{})
 assert not cache.last_hit and len(model.calls)==3
 assert model.release_prompt_cache.call_count==1
 callback.reset_callback_state();cache.invoke(0,1,(ids,None),{})
 assert cache.last_hit and len(model.calls)==4
 cache.close();callback.reset_callback_state()


def test_equal_or_shorter_prefix_is_not_reused(tmp_path):
 from rkllama.api.prefix_cache import PrefixCache
 from rkllama.api import callback
 model=fake_model();cache=PrefixCache(model,tmp_path)
 callback.reset_callback_state();cache.invoke(0,1,(list(range(300)),None),{})
 callback.reset_callback_state();cache.invoke(0,1,(list(range(268)),None),{})
 assert not cache.last_hit and len(model.calls)==4
 cache.close();callback.reset_callback_state()


def test_oversize_snapshot_disables_repeated_priming(tmp_path):
 from rkllama.api.prefix_cache import PrefixCache
 from rkllama.api import callback
 model=fake_model();cache=PrefixCache(model,tmp_path,max_bytes=2);ids=list(range(300))
 callback.reset_callback_state();cache.invoke(0,1,(ids,None),{})
 assert cache.disabled and not cache.prefix and not cache.path.exists()
 callback.reset_callback_state();cache.invoke(0,1,(ids,None),{})
 assert len(model.calls)==3
 cache.close();callback.reset_callback_state()


def test_failed_full_clear_marks_worker_state_unsafe(tmp_path):
 from rkllama.api.prefix_cache import PrefixCache
 model=fake_model();model.rkllm_clear_kv_cache.return_value=-1;cache=PrefixCache(model,tmp_path)
 with pytest.raises(RuntimeError):cache.invoke(0,1,(list(range(300)),None),{})
 assert cache.tainted and not model.calls


def test_abort_during_priming_does_not_start_another_native_call(tmp_path):
 from rkllama.api.prefix_cache import PrefixCache
 from rkllama.api import callback
 model=fake_model();cache=PrefixCache(model,tmp_path);cancelled=[False]
 original=model._run
 def run(*args):
  result=original(*args);cancelled[0]=True;return result
 model._run=run;cache.cancelled=lambda:cancelled[0]
 callback.reset_callback_state();assert cache.invoke(0,1,(list(range(300)),None),{})==-1
 assert len(model.calls)==1 and not cache.prefix and callback.suppress_output is False
 cache.close();callback.reset_callback_state()


def test_bypass_request_does_not_inherit_prior_priming_usage():
 from rkllama.api.rkllm import RKLLM
 from rkllama.api.classes import RKLLMInputType,RKLLMInferMode
 model=RKLLM.__new__(RKLLM)
 model.prefix_cache=SimpleNamespace(last_hit=True,last_priming_metrics=[268,1,10,1],_release_clear=Mock())
 model._run=Mock(return_value=0)
 model.run(RKLLMInferMode.RKLLM_INFER_GENERATE,RKLLMInputType.RKLLM_INPUT_PROMPT,('hello',None),{})
 assert model.prefix_cache.last_priming_metrics==[] and model.prefix_cache.last_hit is False


def test_abort_is_retried_until_native_thread_is_quiescent():
 from test_native_review_regressions import load_worker
 worker=load_worker();model=SimpleNamespace(abort=Mock());state={'alive':True,'joins':0}
 def join(timeout):
  state['joins']+=1
  if state['joins']==3:state['alive']=False
 thread=SimpleNamespace(is_alive=lambda:state['alive'],join=join)
 assert worker.abort_and_join(model,thread,timeout=1)
 assert model.abort.call_count==3 and state['joins']==3


def test_cancellation_at_load_transition_blocks_visible_native_call(tmp_path):
 from rkllama.api.prefix_cache import PrefixCache
 from rkllama.api import callback
 model=fake_model();cache=PrefixCache(model,tmp_path);cancelled=[False];cache.cancelled=lambda:cancelled[0]
 def load(*args):cancelled[0]=True;return 0
 model.rkllm_load_prompt_cache=load
 callback.reset_callback_state();assert cache.invoke(0,1,(list(range(300)),None),{})==-1
 assert len(model.calls)==1 and not cache.prefix
 cache.close();callback.reset_callback_state()


def test_normal_inference_cleanup_preserves_the_owned_prefix_slot():
 from test_native_review_regressions import load_worker
 from queue import Queue
 worker=load_worker();manager=worker.WorkerManager.__new__(worker.WorkerManager)
 queue=Queue();manager.workers={'test':SimpleNamespace(task_queue=queue)}
 manager.send_task=Mock(return_value='pipe')
 assert manager.inference('test',[1,2],None,{})=='pipe'
 task=queue.get_nowait()
 assert task[1]==worker.WORKER_TASK_CLEAR_CACHE and task[4] is True


def test_post_inference_clear_releases_state_without_deleting_snapshot():
 from rkllama.api.rkllm import RKLLM
 model=RKLLM.__new__(RKLLM)
 model.prefix_cache=SimpleNamespace(_release_clear=Mock(),invalidate=Mock())
 assert model.clear_cache(preserve_prefix=True)==0
 model.prefix_cache._release_clear.assert_called_once();model.prefix_cache.invalidate.assert_not_called()
 model.clear_cache();model.prefix_cache.invalidate.assert_called_once()


def test_cache_abort_waits_for_noninterruptible_prefill_to_quiesce():
 from test_native_review_regressions import load_worker
 from unittest.mock import patch as mock_patch
 worker=load_worker();model=SimpleNamespace(abort=Mock(),prefix_cache=object());state={'joins':0,'alive':True}
 def join(timeout):
  state['joins']+=1
  if state['joins']==2:state['alive']=False
 thread=SimpleNamespace(is_alive=lambda:state['alive'],join=join)
 with mock_patch.object(worker.time,'monotonic',side_effect=[0,0,40,40,40]):
  assert worker.abort_and_join(model,thread)
 assert state['joins']==2


def test_unquiescent_worker_exits_without_waiting_on_native_threads():
 from test_native_review_regressions import load_worker
 from unittest.mock import patch as mock_patch
 worker=load_worker()
 with mock_patch.object(worker.os,'_exit',side_effect=SystemExit) as exit_process:
  with pytest.raises(SystemExit):worker.terminate_unquiescent_worker()
 exit_process.assert_called_once_with(1)
