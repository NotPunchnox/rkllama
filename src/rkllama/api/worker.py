import logging
import psutil
import rkllama.config
import time
import threading
import random
from multiprocessing import Process, Queue
from datetime import datetime, timedelta
from.model_utils import get_model_size, get_encoder_model_path, get_property_modelfile
from .classes import *
from .callback import *
    
from operator import attrgetter


logger = logging.getLogger("rkllama.worker")

# Worker variables
WORKER_TASK_UNLOAD_MODEL = "UNLOAD"
WORKER_TASK_EMBEDDING = "EMBEDDING"
WORKER_TASK_INFERENCE = "INFERENCE"
WORKER_TASK_VISION_ENCODER = "VISION_ENCODER"
WORKER_TASK_FINISHED = "<RKLLM_TASK_FINISHED>"
WORKER_TASK_ERROR = "<RKLLM_TASK_ERROR>"
WORKER_TASK_ABORT_INFERENCE = "ABORT"
WORKER_TASK_CLEAR_CACHE = "CLEAR_CACHE"
WORKER_TASK_GENERATE_IMAGE = "GENERATE_IMAGE"
WORKER_TASK_GENERATE_SPEECH = "GENERATE_SPEECH"
WORKER_TASK_GENERATE_TRANSCRIPTION = "GENERATE_TRANSCRIPTION"


def run_encoder(model_input, rknn_queue):
    """
    Run the vision encoder to get the image embedding
    Args:
        model_input (tuple): (model_encoder_path, image_path)
        rknn_queue (Queue): Queue to return the image embedding
    Returns:
        np.ndarray: Image embedding
    """

    from .rknnlite import run_vision_encoder

    # Get the arguments for the call
    model_encoder_path, images_source, image_width, image_height = model_input

    # Run the visionencder to get the image embedding
    image_embeddings = run_vision_encoder(model_encoder_path, images_source, image_width, image_height)
    
    # Send the encoded image to the main process
    rknn_queue.put(image_embeddings)



def run_image_generator(model_input, rknn_queue):
    """
    Run the image generator model to get the image
    Args:
        model_input (tuple): (model_encoder_path, image_path)
        rknn_queue (Queue): Queue to return the image embedding
    Returns:
        str: Image
    """

    from .image_generator import generate_image

    # Get the arguments for the call
    model_name, prompt, size, seed, num_inference_steps, guidance_scale = model_input

    # Run the visionencder to get the image embedding
    image = generate_image(model_name, prompt, size, seed, num_inference_steps, guidance_scale)
    
    # Send the encoded image to the main process
    rknn_queue.put(image)


def run_speech_generator(model_input, rknn_queue):
    """
    Run piper generator model to get the audio
    Args:
        model_input (tuple): (model_piper_path,input,voice,response_format,stream_format,volume,length_scale,noise_scale,noise_w_scale,normalize_audio)
        rknn_queue (Queue): Queue to return the audio
    Returns:
        str: Audio
    """

    from .tts import generate_speech

    # Get the arguments for the call
    model_piper_path,input,voice,response_format,stream_format,volume,length_scale,noise_scale,noise_w_scale,normalize_audio = model_input

    # Run the piper
    audio = generate_speech(model_piper_path,input,voice,response_format,stream_format,volume,length_scale,noise_scale,noise_w_scale,normalize_audio)
    
    # Send the audio bytes to the main process
    rknn_queue.put(audio)


def run_transcription_generator(model_input, rknn_queue):
    """
    Run omniasr generator model to get the transcription
    Args:
        model_input (tuple): (model_omniasr_path,file,language)
        rknn_queue (Queue): Queue to return the transcription
    Returns:
        str: Transcription
    """

    from .stt import generate_transcription

    # Get the arguments for the call
    model_omniasr_path,file,language = model_input

    # Run the omniasr
    audio = generate_transcription(model_omniasr_path,file,language)
    
    # Send the text transcription to the main process
    rknn_queue.put(audio)



# RKLLM Worker 
def run_rkllm_worker(name, task_queue: Queue, result_queue: Queue, model_path, model_dir, options=None, lora_model_path = None, prompt_cache_path = None, base_domain_id = 0):
    
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

    except Exception as e:
        logger.error(f"Failed creating the worker for model '{name}': {str(e)}")
        # Announce the creation of the RKLLM model in memory
        result_queue.put(WORKER_TASK_ERROR)
        return

    # Loop to wait for tasks
    while True:

        try:

            # Get the instruction to the worker
            task,inference_mode, model_input_type, model_input = task_queue.get()

            if task == WORKER_TASK_UNLOAD_MODEL:
                logger.info(f"Unloading model {name}...")
                # Unload the model
                model_rkllm.release()

                # Exit the loop of the worker to finish the process
                break
            
            elif task == WORKER_TASK_ABORT_INFERENCE:
                logger.info(f"Aborting inference for model {name}...")
                # Abort the inference of the model
                model_rkllm.abort()

            elif task == WORKER_TASK_CLEAR_CACHE:
                logger.info(f"Clearing KV cache for model {name}...")
                # CLear the cache of the model
                model_rkllm.clear_cache()

            elif task == WORKER_TASK_INFERENCE:
                logger.info(f"Running inference for model {name}...")
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

                # CLear the cache after inference
                model_rkllm.clear_cache()

                # Send final signal of the inference
                result_queue.put(WORKER_TASK_FINISHED)

            elif task == WORKER_TASK_EMBEDDING:
                logger.info(f"Running embedding for model {name}...")
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
            
            elif task == WORKER_TASK_VISION_ENCODER:
                logger.info(f"Running vision encoder for model {name}...")
                # Run the vision encoder to get the image embedding
                rknn_queue = Queue()

                # Define the process for the encoder
                rknn_process = Process(target=run_encoder, args=(model_input,rknn_queue,))

                # Start the encoder worker
                rknn_process.start() 

                # Get the encoded image from the queue
                img_encoded = rknn_queue.get(timeout=60)  # Timeout after 60 seconds

                # Terminate the process encoder after use
                rknn_process.terminate()

                # Send the encoded image 
                result_queue.put(img_encoded)

            else:
                result_queue.put(f"Unknown task: {task}")
                # Send final signal of the inference
                result_queue.put(WORKER_TASK_FINISHED)

        except Exception as e:
            logger.error(f"Failed executing task the worker for model '{name}' for task '{task}': {str(e)}")
            # Announce the creation of the RKLLM model in memory
            result_queue.put(WORKER_TASK_ERROR)



# RKNN Process 
def run_rknn_process(name, task, model_input, result_queue: Queue):
    
    try:

        if task == WORKER_TASK_GENERATE_IMAGE:
            logger.info(f"Running image generator for model {name}...")
            # Run the vision encoder to get the image embedding
            rknn_queue = Queue()

            # Define the process for the encoder
            rknn_process = Process(target=run_image_generator, args=(model_input,rknn_queue,))

            # Start the encoder worker
            rknn_process.start() 

            # Get the encoded image from the queue
            img = rknn_queue.get(timeout=300)  # Timeout after 300 seconds

            # Terminate the process encoder after use
            rknn_process.terminate()

            # Send the image 
            result_queue.put(img)

        elif task == WORKER_TASK_GENERATE_SPEECH:
            logger.info(f"Running speech generator for model {name}...")
            # Run piper
            rknn_queue = Queue()

            # Define the process for piper
            rknn_process = Process(target=run_speech_generator, args=(model_input,rknn_queue,))

            # Start the piper worker
            rknn_process.start() 

            # Get the audio from the queue
            audio = rknn_queue.get(timeout=300)  # Timeout after 300 seconds

            # Terminate the process piper after use
            rknn_process.terminate()

            # Send the audio 
            result_queue.put(audio)

        elif task == WORKER_TASK_GENERATE_TRANSCRIPTION:
            logger.info(f"Running transcription generator for model {name}...")
            # Run omniasr
            rknn_queue = Queue()

            # Define the process for omniasr
            rknn_process = Process(target=run_transcription_generator, args=(model_input,rknn_queue,))

            # Start the omniasr worker
            rknn_process.start() 

            # Get the text from the queue
            text = rknn_queue.get(timeout=300)  # Timeout after 300 seconds

            # Terminate the process omniasr after use
            rknn_process.terminate()

            # Send the text 
            result_queue.put(text)

        else:
            result_queue.put(f"Unknown task: {task}")
            # Send final signal of the inference
            result_queue.put(WORKER_TASK_FINISHED)

    except Exception as e:
        logger.error(f"Failed executing task the rknn process for model '{name}' for task '{task}': {str(e)}")
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


    def get_available_base_domain_id(self, reverse_order=False) -> int | None:
        """
        Returns the smallest available integer between 1 and 10
        that is not already used as 'base_domain_id' in the current list of worker process.
        If all numbers from 1 to 10 are taken, returns None.

        Args:
            reverse_order (bool): If true, search from the highest to the lowest.
        
        Returns:
            int | None: The available base_domain_id or None if all are taken.
        """
        # Get all used base domain ids
        used_base_domain_ids = [self.workers[model].worker_model_info.base_domain_id for model in self.workers.keys()]

        # Get the max id of a domain base:
        max_domain_id = int(rkllama.config.get("model", "max_number_models_loaded_in_memory"))

        if reverse_order:
            # CHeck fir available from the highest to the lowest
            candidates_range = range(max_domain_id, 0, -1)
        else:
            # CHeck first available from the lowest to the highest  
            candidates_range = range(1, max_domain_id)
        
        # CHeck fir available
        for candidate in candidates_range:
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
            base_domain_id = self.get_available_base_domain_id(reverse_order=True)

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
        return (psutil.virtual_memory().available + psutil.virtual_memory().free) > (model_size * 1.20) # Include 20% more memory required than the model size
    

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
            self.workers[model_name].worker_model_info.expires_at = datetime.now() + timedelta(minutes=int(rkllama.config.get("model", "max_minutes_loaded_in_memory")),)



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


    def clear_cache_worker(self, model_name):
        """
        Clear the KV chache of a model worker
        
        Args:
            model_name (str): Workers to clear cache.

        """
        if model_name in self.workers.keys():
            # Get the queue of tasks of the worker

            # Send the abort task of the model if currently is running some inference
            self.workers[model_name].task_q.put((WORKER_TASK_CLEAR_CACHE,None,None,None))


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
            
    
    def multimodal(self, model_name, prompt_input, images):
        """
        Send a inference task to the corresponding model worker for multimodal input
        
        Args:
            model_name (str): Model name to invoke
            prompt_input (str): Input of the model
            image_embed (np.ndarray): Image embedding
            n_image_tokens (int): Number of image tokens
            image_width (int): Width of the image
            image_height (int): Height of the image

        """

        if model_name in self.workers.keys():

            # Get the path of the vision encoder model
            model_encoder_path = get_encoder_model_path(model_name)

            # Check if the encoder model is available
            if model_encoder_path is None:
                # No vision encoder model available for this RKLLM model
                raise RuntimeError(f"No encoder model (.rknn) found for : {model_name}")

            # Get properties of the encoder model
            image_width = int(get_property_modelfile(model_name, 'IMAGE_WIDTH', rkllama.config.get_path("models"))) 
            image_height = int(get_property_modelfile(model_name, 'IMAGE_HEIGHT', rkllama.config.get_path("models"))) 
            n_image_tokens = int(get_property_modelfile(model_name, 'N_IMAGE_TOKENS', rkllama.config.get_path("models"))) 
            num_images = len(images)

            # Prepare the image input embed for multimodal
            image_embed  =  self.get_images_embed(model_name, model_encoder_path, images, image_width, image_height)

            # Check if the image was encoded correctly
            if image_embed is None:
                # Error encoding the image. Return
                raise RuntimeError(f"Unexpected error encoding image for model : {model_name}")
            
            # Prepare all the inputs for the multimodal inference
            model_input = (prompt_input, image_embed, n_image_tokens, image_width, image_height, num_images)
            
            # Send the inference task
            self.send_task(model_name, (WORKER_TASK_INFERENCE,RKLLMInferMode.RKLLM_INFER_GENERATE, RKLLMInputType.RKLLM_INPUT_MULTIMODAL, model_input))


    def get_images_embed(self, model_name, model_encoder_path, images, image_width, image_height) -> None:
        """
        Send a vision encoder task to the corresponding model worker
        
        Args:
            model_name (str): Model name to invoke
            model_encoder_path (str): Path of the vision encoder model
            images (list): List of image paths/base64/urls
            image_width (int): Width of the image
            image_height (int): Height of the image
        """
        if model_name in self.workers.keys():
            
            # Get model encoder size
            model_encoder_size = os.path.getsize(model_encoder_path)
            # Check if available meory in server for encoder
            if not self.is_memory_available_for_model(model_encoder_size):
                # Unload the oldest model until memory avilable
                self.unload_oldest_models_from_memory(model_encoder_size)

            # Prepare the input for the vision encoder
            model_input = (model_encoder_path, images, image_width, image_height)

            # Send the Encoder task of the image
            self.send_task(model_name, (WORKER_TASK_VISION_ENCODER,None, None, model_input))  

            # Wait to confirm output of the image encoder
            image_embed  = self.workers[model_name].result_q.get(timeout=60)  # Timeout after 60 seconds

            if isinstance(image_embed, str) and image_embed ==  WORKER_TASK_ERROR:
                # Error ENcoding the image. Return
                return None
     
            # Return the image encoded
            return image_embed;       



    def generate_image(self, model_name,model_dir, prompt, size, num_images, seed, num_inference_steps, guidance_scale) -> None:
        """
        Send a generate image task to the corresponding model worker
        
        Args:
            model_name (str): Worker name to send the task.
            model_dir (str): Model directory name to invoke
            prompt (str): Prompt to generate the image
            stream (bool): If true, stream the response
            size (str): Size of the image
            num_images (int): Number of images to generate
            seed (int): Seed for the random number generator
            num_inference_steps (int): Number of inference steps
            guidance_scale (float): Guidance scale for the generation
        """

        # List to store the generated images
        image_list = []

        # Loop over the number of images to generate
        for image in range(num_images):

            if image > 1:
                # For the next images, use a different seed
                seed = random.randint(1, 99)

            # Prepare the input for the vision encoder
            model_input = (model_dir, prompt, size, seed, num_inference_steps, guidance_scale)

            # Result queue for the RKNN process
            result_queue = Queue()

            # Send the Encoder task of the image
            run_rknn_process(model_name, WORKER_TASK_GENERATE_IMAGE,model_input,result_queue)  

            # Wait to confirm output of the image 
            image_base  = result_queue.get(timeout=300)  # Timeout after 60 seconds

            if isinstance(image_base, str) and image_base ==  WORKER_TASK_ERROR:
                # Error ENcoding the image. Return
                return None

            # Add the image to the list
            image_list.append(image_base)    
    
        # Return the image
        return image_list;   


    def generate_speech(self, model_name, model_dir, input,voice,response_format,stream_format,volume,length_scale,noise_scale,noise_w_scale,normalize_audio) -> None:
        """
        Send a generate speech task to the corresponding model worker
        
        Args:
            model_name (str): Worker name to send the task.
            model_dir (str): Model directory name to invoke
 
        """

        # Prepare the input for piper
        model_input = (model_dir, input,voice,response_format,stream_format,volume,length_scale,noise_scale,noise_w_scale,normalize_audio)

        # Result queue for the RKNN process
        result_queue = Queue()

        # Send the Encoder task of the Speech
        run_rknn_process(model_name, WORKER_TASK_GENERATE_SPEECH,model_input,result_queue)  

        # Wait to confirm output of the image 
        audio  = result_queue.get(timeout=300)  # Timeout after 60 seconds

        if isinstance(audio, str) and audio ==  WORKER_TASK_ERROR:
            # Error Generating the speech. Return
            return None

        # Return the audio
        return audio
    
    def generate_transcription(self, model_name, model_dir, file, language, response_format) -> None:
        """
        Send a generate transcription task to the corresponding model worker
        
        Args:
            model_name (str): Worker name to send the task.
            model_dir (str): Model directory name to invoke
 
        """

        # Prepare the input for omniasr
        model_input = (model_dir, file, language)

        # Result queue for the RKNN process
        result_queue = Queue()

        # Send the inference task of the Transcription
        run_rknn_process(model_name, WORKER_TASK_GENERATE_TRANSCRIPTION,model_input,result_queue)  

        # Wait to confirm output of the image 
        text  = result_queue.get(timeout=300)  # Timeout after 60 seconds

        if isinstance(text, str) and text ==  WORKER_TASK_ERROR:
            # Error Generating the transcription. Return
            return None

        # Return the transcription
        return text
    

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
        self.expires_at = datetime.now() + timedelta(minutes=int(rkllama.config.get("model", "max_minutes_loaded_in_memory")))
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
        self.process = Process(target=run_rkllm_worker, args=(self.worker_model_info.model, self.task_q, self.result_q, model_path, model_dir, options, lora_model_path, prompt_cache_path, base_domain_id))

        # Start the worker
        self.process.start() 

        # Wait to confirm initialization
        creation_status = self.result_q.get(timeout=60)  # Timeout after 60 seconds

        if creation_status == WORKER_TASK_ERROR:
            # Error loading the RKLLM Model. Wait for the worker to exit
            self.process.terminate()
            return False
        
        # Success loading the model
        return True
        
