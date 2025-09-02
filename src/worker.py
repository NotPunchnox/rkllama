
from .classes import *
from .callback import *
import logging
from multiprocessing import Process, Queue
import time
import threading
from datetime import datetime, timedelta
from.model_utils import get_model_size
import psutil
import config
from operator import attrgetter


logger = logging.getLogger("rkllama.worker")

# Worker variables
WORKER_TASK_UNLOAD_MODEL = "UNLOAD"
WORKER_TASK_EMBEDDING = "EMBEDDING"
WORKER_TASK_INFERENCE = "INFERENCE"
WORKER_TASK_FINISHED = "<RKLLM_TASK_FINISHED>"
WORKER_TASK_ERROR = "<RKLLM_TASK_ERROR>"
WORKER_TASK_ABORT_INFERENCE = "ABORT"


# Worker 
def run_worker(name, task_queue: Queue, result_queue: Queue, model_path, model_dir, options=None, lora_model_path = None, prompt_cache_path = None, base_domain_id = 0):
    
    # Initialize individual callback for each worker to prevent error from RKLLM
    from .callback import callback_impl, global_status, global_text,split_byte_data, last_embeddings
    from .rkllm import RKLLM

    # Connect the callback function between Python and C++ independently for each worker
    callback = callback_type(callback_impl)

    # Define the model used by the worker
    try:
        model_rkllm = RKLLM(callback, model_path, model_dir, options, lora_model_path, prompt_cache_path, base_domain_id)
    
        # Announce the creation of the RKLLM model failed
        result_queue.put(WORKER_TASK_FINISHED)
        
        while True:
            
            # Get the instruction to the worker
            task ,inference_mode, model_input_type, model_input = task_queue.get()

            if task == WORKER_TASK_UNLOAD_MODEL:
                logger.info(f"Unloading model {name}...")
                # Unload the model
                model_rkllm.release()

                # Exit the loop of the worker to finish the process
                break
            
            elif task == WORKER_TASK_ABORT_INFERENCE:

                # Abort the inference of the model
                model_rkllm.abort()

            elif task == WORKER_TASK_INFERENCE:

                # Run inference
                thread_model = threading.Thread(target=model_rkllm.run, args=(inference_mode, model_input_type, model_input,))
                thread_model.start()
                
                # Looping until execution of the thread
                thread_finished = False
                while not thread_finished:
                    tokens_processed = False
                    while len(global_text) > 0:
                        tokens_processed = False
                        token = global_text.pop(0)
                        result_queue.put(token)

                    # Update status of the thread    
                    thread_model.join(timeout=0.005)
                    thread_finished = not thread_model.is_alive()
                    
                    # If inference not started yet, wait some time to start.
                    if not tokens_processed:
                        time.sleep(0.01)

                # Send final signal of the inference
                result_queue.put(WORKER_TASK_FINISHED)

            elif task == WORKER_TASK_EMBEDDING:

                # Run inference
                thread_model = threading.Thread(target=model_rkllm.run, args=(inference_mode, model_input_type, model_input,))
                thread_model.start()
                
                # Looping until execution of the thread finished
                thread_finished = False
                while not thread_finished:
                    # Update status of the thread    
                    thread_model.join(timeout=0.005)
                    thread_finished = not thread_model.is_alive()

                if last_embeddings:
                    # Send the embedding shapes of the input
                    result_queue.put(last_embeddings[0])
            
            else:
                result_queue.put(f"Unknown task: {task}")
                # Send final signal of the inference
                result_queue.put(WORKER_TASK_FINISHED)

    except Exception as e:
        logger.error(f"Failed creating the worker for model '{name}': {str(e)}")
        # Announce the creation of the RKLLM model in memory
        result_queue.put(WORKER_TASK_ERROR)


# Class to manage the workers for RKLLM models
class WorkerManager:
    def __init__(self):
        self.workers = {}  #  (name -> Worker)

        # Start the monitor of running models
        self.start_models_monitor()

    def start_models_monitor(self, interval=60):
        """
        Start a threat to monitor expired models to unload them from memory
        
        Args:
            interval: Interval between check
        """
        def execute():
            while True:
                try:
                    # Call the process to unload expired models
                    self.unload_expired_models()
                    # Wait for the next execution
                    time.sleep(interval)  # Check every 60 seconds expired models
                except Exception as e:
                    logger.error(f"Exception in monitor models: {e}")
 
        # Iniciar el hilo como daemon (no bloquea al final del programa)
        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        logger.info("Models Monitor running.")

    
    def unload_expired_models(self) -> int | None:
        """
        Unload/stop workers for expired models
        """
        # Get all expired models
        expired_models = [ model for model in self.workers.keys() if datetime.now() > self.workers[model].worker_model_info.expires_at ]
        
        # Unload/stop the expired model
        for model_name in expired_models:
            logger.info(f"Detected expired model: {model_name}")
            self.stop_worker(model_name)


    def get_available_base_domain_id(self) -> int | None:
        """
        Returns the smallest available integer between 1 and 10
        that is not already used as 'base_domain_id' in the current list of worker process.
        If all numbers from 1 to 10 are taken, returns None.
        """
        # Get all used base domain ids
        used_base_domain_ids = [self.workers[model].worker_model_info.base_domain_id for model in self.workers.keys()]

        # CHeck fir available
        for candidate in range(1, config.get("model", "max_number_models_loaded_in_memory")):
            if candidate not in used_base_domain_ids:
                return candidate
        return None


    def exists_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model with the given model_name exists in the dict of workers
        Args:
            model_name (str): Model name to check if already loaded in memory.
        
        """
        return model_name in self.workers.keys()


    def add_worker(self, model_name, model_path, model_dir, options=None, lora_model_path = None, prompt_cache_path = None) -> bool:
        """
        Add a process worker to run inferences call from a specific model
        
        Args:
            model_name (str): model name to load in memory
        """
        if model_name not in self.workers.keys():

            # Get the available domain id for the RKLLM process
            base_domain_id = self.get_available_base_domain_id()

            # Add the worker to the dictionary of workers
            worker_model = Worker(model_name,base_domain_id)

            # Check if available meory in server
            if not self.is_memory_available_for_model(worker_model.worker_model_info.size):
                # Unload the oldest model until memory avilable
                self.unload_oldest_models_from_memory(worker_model.worker_model_info.size)

            # Initializae de worker/model
            model_loaded = worker_model.create_worker_process(base_domain_id, model_path, model_dir, options, lora_model_path, prompt_cache_path)

            # Check the load of the model
            if not model_loaded:
                # Error loading the model
                return False
            else:    
                # Add the worker to the dictionary of workers
                self.workers[model_name] = worker_model
                logger.info(f"Worker for model {model_name} created and running...")
                return True


    def unload_oldest_models_from_memory(self, memory_required):
        """
        Unload the oldest models from meory
        Args:
            memory_required (int) -> Size of memory need by the model to load
        """
        # From the dictionary of workers, we create an array of worker info that holds the size of each one
        worker_models_info = [ self.workers[model].worker_model_info for model in self.workers.keys() ]

        # Loop over the array by the oldest worker model
        for worker_model_info in sorted(worker_models_info, key=attrgetter('last_call')):
            logger.info(f"Unloading model {worker_model_info.model} to gain free memory (at least {memory_required})")
            # Stop the first oldest modelin memory
            self.stop_worker(worker_model_info.model)

            # Wait a second to refresh memory system
            time.sleep(1)

            # CHeck if now memory available for the new model to load 
            if self.is_memory_available_for_model(memory_required):
                break


    def is_memory_available_for_model(self, model_size) -> bool:
        """
        Check if exist memory available for model load
        Args:
            model_size (int) -> Size of the model to load
        """
        return (psutil.virtual_memory().available + psutil.virtual_memory().free) > model_size
    

    def send_task(self, model_name, task):
        """
        Send a task to execute for the RKLLM model
        Args:
            model_name (str): Worker name to send the task.
            task (tuple (name_task,args)): Task to send to the worker 

        """
        if model_name in self.workers:
            # Send the TASK to the model with the communication queue of the model 
            self.workers[model_name].task_q.put(task)

            # Update the worker model info with the invocation
            self.workers[model_name].worker_model_info.last_call = datetime.now()
            self.workers[model_name].worker_model_info.expires_at = datetime.now() + timedelta(minutes=config.get("model", "max_minutes_loaded_in_memory"),)



    def get_result(self, model_name):
        """
        Get the result of a task executed for the RKLLM model
        
        Args:
            model_name (str): Worker name to get the response.

        Returns:
            Queue: Queue for the worker where the response is stored.
        """
        if model_name in self.workers:
            # Get the queue of the responses of the worker
            return self.workers[model_name].result_q
        return None


    def stop_worker(self, model_name):
        """
        Stop/Unload a model worker
        
        Args:
            model_name (str): Workers to unload.

        """
        if model_name in self.workers.keys():
            # Get the queue of tasks of the worker

            # Send the abort task of the model if currently is running some inference
            self.workers[model_name].task_q.put((WORKER_TASK_ABORT_INFERENCE,None,None,None))

            # Send the unload task of the model
            self.workers[model_name].task_q.put((WORKER_TASK_UNLOAD_MODEL,None,None,None))

            # Wait for unload
            self.workers[model_name].process.join()
            logger.info(f"Worker {model_name} stopped...")

            # Remove the worker from the dictionary
            del self.workers[model_name]

    def stop_all(self):
        """
        Send a inference task to the corresponding model worker
        """
        # Loop over all the workers to stop/unload
        for model_name in list(self.workers.keys()):
            self.stop_worker(model_name)


    def inference(self, model_name, model_input):
        """
        Send a inference task to the corresponding model worker
        
        Args:
            model_name (str): Model name to invoke
            model_input (str): Input of the model

        """
        if model_name in self.workers.keys():
            # Send the inference task
            self.send_task(model_name, (WORKER_TASK_INFERENCE,RKLLMInferMode.RKLLM_INFER_GENERATE, RKLLMInputType.RKLLM_INPUT_TOKEN, model_input))

    
    def embedding(self, model_name, model_input):
        """
        Send a prepare embedding task to the corresponding model worker
        
        Args:
            model_name (str): Model name to invoke
            model_input (str): Input of the model

        """
        if model_name in self.workers.keys():
            # Send the inference task
            self.send_task(model_name, (WORKER_TASK_EMBEDDING,RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER, RKLLMInputType.RKLLM_INPUT_TOKEN, model_input))        
            
    
    def get_finished_inference_token(self):
        """
        Return the finish token for inference task
        
        Returns:
            str: Token for finished inference.
        """
        return WORKER_TASK_FINISHED
                


# Class to manage the information for running RKLLM models
class WorkerModelInfo:
    def __init__(self, model_name, base_domain_id):
        self.model = model_name
        self.size = get_model_size(model_name)
        self.expires_at = datetime.now() + timedelta(minutes=config.get("model", "max_minutes_loaded_in_memory"))
        self.loaded_at = datetime.now()
        self.base_domain_id = base_domain_id
        self.last_call = datetime.now()
                        
      
# Class to manage the information for running RKLLM models
class Worker:
    def __init__(self, model_name, base_domain_id):
        self.worker_model_info = WorkerModelInfo(model_name=model_name, base_domain_id=base_domain_id)
        self.process = None
        self.task_q = Queue()
        self.result_q = Queue()


    def create_worker_process(self, base_domain_id, model_path, model_dir, options=None, lora_model_path = None, prompt_cache_path = None) -> bool:
        """
        Creates the process of the worker
        """

        # Define the process for the worker
        self.process = Process(target=run_worker, args=(self.worker_model_info.model, self.task_q, self.result_q, model_path, model_dir, options, lora_model_path, prompt_cache_path, base_domain_id))

        # Start the worker
        self.process.start() 

        # Wait to confirm initialization
        creation_status = self.result_q.get()

        if creation_status == WORKER_TASK_ERROR:
            # Error loading the RKLLM Model. Wait for the worker to exit
            self.process.join()
            return False
        
        # Success loading the model
        return True
        
