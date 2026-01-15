import numpy as np
import soundfile as sf
import logging
from typing import Tuple
import os
import whisper
import time
import soxr
import webrtcvad
import configparser
from transformers import WhisperTokenizerFast
from rknnlite.api import RKNNLite

logger = logging.getLogger("rkllama.audio.whisper")

class WhisperSTTModelRKNN:
    def __init__(
        self,
        model_path: str,
    ):

        # Get encoder, decoder, adecoder with past and tokenizer 
        encoder, decoder, decoder_with_past, tokenizer, configuration = self.find_model_files(model_path)

        # Prepare the RKNN runtime model
        # Encoder
        self.encoder_rknn = RKNNLite(verbose=False)
        self.encoder_rknn.load_rknn(encoder)
        self.encoder_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) # Best performance ( 25% faster in whisper large-v3-turbo)
        # Decoder
        self.decoder_rknn = RKNNLite(verbose=False)
        self.decoder_rknn.load_rknn(decoder)
        self.decoder_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        # Decoder with past
        self.decoder_with_past_rknn = RKNNLite(verbose=False)
        self.decoder_with_past_rknn.load_rknn(decoder_with_past)
        self.decoder_with_past_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

        # Save the tokenizer of the  model
        self.tokenizer = WhisperTokenizerFast.from_pretrained(tokenizer)

        # Save the tokenizer of the  model
        self.configuration = configparser.ConfigParser()
        self.configuration.read(configuration)
    
        # Load Inputs and Outputs of the models
        self.load_inputs_outputs_from_rknn_models()


    def find_model_files(self, dir_path: str) -> Tuple[str, str]:
        """
        Find exactly one .rknn for encoder, one .rknn for decoder and one .rknn for decoder with past, tokeniziner directory  and confihguration for whisper model.

        Returns:
            (rknn_path, rknn_path, rknn_path, tokenizer_path, configuration)

        Raises:
            FileNotFoundError: if a required file type is missing
            ValueError: if more than one file of a type is found
        """

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        
        # Search for the encoder model
        encoder_rknn = self.get_rknn_model_path(os.path.join(dir_path, "encoder"))

        # Search for the decoder model
        decoder_rknn = self.get_rknn_model_path(os.path.join(dir_path, "decoder"))

        # Search for the decoder with past model
        decoder_with_past_rknn = self.get_rknn_model_path(os.path.join(dir_path, "decoder_with_past"))

        # Search for the tokenizer of the model
        tokenizer = os.path.join(dir_path, "tokenizer")

        # Search for the configuration of the model
        configuration = os.path.join(dir_path, "whisper.ini")

        # Return the files    
        return ( encoder_rknn, decoder_rknn, decoder_with_past_rknn, tokenizer, configuration)


    def get_rknn_model_path(self, dir_path) -> str:
        """
        Return the RKNN filename path in a directory
        """

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        model_rknn = None

        # Search for the rknn model
        for name in os.listdir(dir_path):
            path = os.path.join(dir_path, name)
            if not os.path.isfile(path):
                # Skip dictionaries
                continue

            # Check the current file in the directoty loop
            ext = os.path.splitext(name)[1].lower()
            if ext == ".rknn":
                if not model_rknn:
                    model_rknn = path
                    break

        # Return the model    
        return model_rknn


    def load_inputs_outputs_from_rknn_models(self):
        """
        Load the inputs and outputs from the RKNN models loaded into memory in the expected order by the model (REQUIRED)

        """

        # Get inputs and outputs from encoder
        encoder_runtime = self.encoder_rknn.rknn_runtime
        n_input_encoder, n_output_encoder = encoder_runtime.get_in_out_num()

        # Get inputs and outputs from decoder without past
        decoder_runtime = self.decoder_rknn.rknn_runtime
        n_input_decoder, n_output_decoder = decoder_runtime.get_in_out_num()

        # Get inputs and outputs from decoder with past
        decoder_with_past_runtime = self.decoder_with_past_rknn.rknn_runtime
        n_input_decoder_with_past, n_output_decoder_with_past = decoder_with_past_runtime.get_in_out_num()

        # Get the inputs and outputs name in the order expected by the model
        self.decoder_rknn_order_inputs_names = [ decoder_runtime.get_tensor_attr(i, is_output=False).name.decode("utf-8") for i in range(n_input_decoder) ]
        self.decoder_rknn_order_output_names = [ decoder_runtime.get_tensor_attr(i, is_output=True).name.decode("utf-8") for i in range(n_output_decoder) ]

        # Get the inputs and outputs name in the order expected by the model with past
        self.decoder_with_past_rknn_order_inputs_names = [ decoder_with_past_runtime.get_tensor_attr(i, is_output=False).name.decode("utf-8") for i in range(n_input_decoder_with_past) ]
        self.decoder_with_past_rknn_order_output_names = [ decoder_with_past_runtime.get_tensor_attr(i, is_output=True).name.decode("utf-8") for i in range(n_output_decoder_with_past) ]

        # Get the expeted MEL by the model
        self.n_mel = encoder_runtime.get_tensor_attr(0, is_output=False).dims[1]

        # Get the max tokens allowed by the decoder with past
        self.max_tokens = decoder_with_past_runtime.get_tensor_attr(1, is_output=False).dims[2]



    def vad_chunks_for_whisper(self, wav_file):
        
        # Get configuration variables
        sample_rate=int(self.configuration["DEFAULT"]["SAMPLE_RATE"])
        frame_ms=int(self.configuration["VAD"]["FRAME_MS"])
        vad_mode=int(self.configuration["VAD"]["VAD_MODE"])
        max_silence_ms=int(self.configuration["VAD"]["MAX_SILENCE_MS"])
        min_chunk_ms=int(self.configuration["VAD"]["MIN_CHUNK_MS"])
        pad_ms=int(self.configuration["VAD"]["PAD_MS"])

        # Define the VAD
        vad = webrtcvad.Vad(vad_mode)

        # Load audio
        audio, sr = sf.read(wav_file, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != sample_rate:
            audio = soxr.resample(audio, sr, sample_rate)
            sr = sample_rate

        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        # PCM16 only for VAD
        audio_i16 = (audio * 32768).astype(np.int16)

        frame_size = int(sr * frame_ms / 1000)
        max_silence_frames = max_silence_ms // frame_ms
        min_chunk_frames = min_chunk_ms // frame_ms
        pad = int(sr * pad_ms / 1000)

        chunks = []
        current = []
        silence = 0

        for i in range(0, len(audio_i16) - frame_size, frame_size):
            frame_i16 = audio_i16[i:i + frame_size]
            speech = vad.is_speech(frame_i16.tobytes(), sr)

            if speech:
                current.append(frame_i16)
                silence = 0
            else:
                if current:
                    silence += 1
                    if silence <= max_silence_frames:
                        current.append(frame_i16)
                    else:
                        if len(current) >= min_chunk_frames:
                            chunk = np.concatenate(current).astype(np.float32) / 32768.0
                            chunk = np.pad(chunk, (pad, pad))
                            chunks.append(chunk)

                        current = []
                        silence = 0

        # Flush final
        if current and len(current) >= min_chunk_frames:
            chunk = np.concatenate(current).astype(np.float32) / 32768.0
            chunk = np.pad(chunk, (pad, pad))
            chunks.append(chunk)

        return chunks


    def merge_whisper_token_chunks(self, token_chunks, max_overlap_tokens=80):
        """
        Merge Whisper token chunks BEFORE decoding.
        Removes duplicated token overlaps efficiently.

        token_chunks: List[List[int]]
        returns: List[int]
        """

        merged = []

        for tokens in token_chunks:
            if not merged:
                merged.extend(tokens)
                continue

            max_k = min(len(tokens), len(merged), max_overlap_tokens)
            overlap = 0

            # Buscar mayor solapamiento exacto
            for k in range(max_k, 0, -1):
                if merged[-k:] == tokens[:k]:
                    overlap = k
                    break

            merged.extend(tokens[overlap:])

        return merged


    def nchw_to_nhwc(self, x: np.ndarray) -> np.ndarray:
        """
        COnvert NCHW → NHWC only if tensor is 4D.
        Ignore inputs like [1,1] or others.
        """
        # CHeck dimensions
        if x.ndim != 4:
            return x

        # (N, C, H, W) → (N, H, W, C)
        return np.transpose(x, (0, 2, 3, 1))

    
    
    def decode_tokens(self, tokens):
        """
        Decode the tokens for transcription
        
        tokens [int]: Array of tokens
        """

        # Decode the tokens
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Return the transcription
        return text
        

    def remove_pad_from_tensor_dim3(self, tensor, num3):
        """
        Crop tensor from (num1, num2, X, num4) to (num1, num2, num3, num4)
        num3 must be <= X
        """
        orig_shape = tensor.shape  # (num1, num2, X, num4)

        if num3 > orig_shape[2]:
            raise ValueError("num3 must be <= original third dimension")
        fixed = tensor[:,:, :num3, :]
        return fixed


    def remove_pad_from_tensor_dim2(self, tensor, num2):
        """
        Crop tensor from (num1, X, num3) to (num1, num2, num3)
        num2 must be <= X
        """
        orig_shape = tensor.shape  # (num1, X, num3)

        if num2 > orig_shape[1]:
            raise ValueError("num2 must be <= original second dimension")

        # Crop only the second dimension
        fixed = tensor[:, :num2, :]

        return fixed


    def init_past(self, instructions, decoder_rknn, hidden_states):
        
        # Set the first input of the model with the instructions
        input_ids = np.array([instructions], dtype=np.int64)

        # Padded the input accorind the dim expected by the decoder without past
        padded_z = np.zeros((input_ids.shape[0], 5), dtype=np.float32)
        padded_z[:, :input_ids.shape[1]] = input_ids
        
        # Call first decoder without past to geenrate logits and first past to the other decoder
        decoder_start = time.time()
        decoder_output = decoder_rknn.inference(inputs=[padded_z, hidden_states], data_format="nchw")
        logger.debug(f"Decoder time: {(time.time() - decoder_start):.2f}s")
        
        # Fix dimension of logits with the original dim of the input without padding
        logits = self.remove_pad_from_tensor_dim2(decoder_output[0],input_ids.shape[1])

        # Construct initial past with the output of the decoder without past
        past = {}
        for i,output in enumerate(self.decoder_rknn_order_output_names):
            if output.startswith("present."):
                new_past_key_name = output.replace('present', 'past_key_values')
                if "decoder" in output:
                    past[new_past_key_name] = self.remove_pad_from_tensor_dim3(decoder_output[i],input_ids.shape[1])
                else:
                    past[new_past_key_name] = decoder_output[i]
        # Return the first logits and past
        return logits, past
        

    def transcribe(self, instructions, wav_file):

        # Split the audio by VAD in chunks
        audios = self.vad_chunks_for_whisper(wav_file)
        logger.debug(f"Number of generated chunks by VAD: {len(audios)}")

        # Loop over every chunk audio to transcribe
        all_tokens = []    
        for audio in audios:    
            
            # Audio → Mel
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels = self.n_mel)
            arr_expanded = np.expand_dims(mel.numpy(), axis=0)
            
            # Encoder
            # Inference RKNN encoder
            encoder_start = time.time()
            hidden_states = self.encoder_rknn.inference(inputs=[arr_expanded], data_format="nchw")[0]
            logger.debug(f"Encoder time: {(time.time() - encoder_start):.2f}s")

            # Decoder without past
            # Initialize the first past with the first decoder without past
            logits, past = self.init_past(instructions, self.decoder_rknn, hidden_states)

            # Decoder with past
            tokens = []
            for step in range(self.max_tokens):
                logger.debug(f"Current loop by decoder with past ={step}")
                
                # Get the token from the logits
                next_token = int(np.argmax(logits[0, -1]))
                logger.debug(f"Generated token = {next_token}")
                

                # Check if EOS token reacheded
                if next_token == int(self.configuration["DEFAULT"]["EOS_TOKEN_ID"]):
                    break

                # Add the token to the list of tokens of the audio chunk    
                tokens.append(next_token)

                # Prepare the input with the last generated token fot the next call of the decoding process
                input_ids = np.array([[next_token]], dtype=np.int64)
                inputs = {"input_ids": input_ids}
                inputs.update(past)

                # Call the decoder with past
                decoder_with_past_start = time.time()
                decoder_output = self.decoder_with_past_rknn.inference(inputs=[self.nchw_to_nhwc(inputs[key]) for key in inputs.keys() ], data_format="nhwc")
                logger.debug(f"Decoder With Past time: {(time.time() - decoder_with_past_start):.2f}s")
                

                # Get the logits for the current inference
                logits = decoder_output[0]
                
                # Update only decoder past
                new_past_rknn = {}
                for i,output in enumerate(self.decoder_with_past_rknn_order_output_names):
                    if output.startswith("present.") and "decoder" in output:
                        new_past_rknn[output.replace('present', 'past_key_values')] = decoder_output[i]

                # encoder past se mantiene
                for k in past:
                    if "encoder" in k:
                        #new_past[k] = past[k]
                        new_past_rknn[k] = past[k]
                
                # Sort inputs for RKNN (Required)
                ordered_past_rknn ={}
                for input in self.decoder_with_past_rknn_order_inputs_names:
                    if input in new_past_rknn.keys():
                        ordered_past_rknn[input] =new_past_rknn[input]
                

                #past = ordered_past
                past = ordered_past_rknn

            # Add all the tokens of the current chunk to the final list of tokens
            all_tokens.append(tokens)
        
        # Merge all the tokens from chunks in the same token list
        merged_tokens = self.merge_whisper_token_chunks(all_tokens)

        # Decode the tokens with the tokenizer
        transcription_text = self.decode_tokens(merged_tokens)

        # Return the transcription
        return transcription_text
    
    def release_rknn_models(self):
        # Release resources from RKNN
        self.encoder_rknn.release()
        self.decoder_rknn.release()
        self.decoder_with_past_rknn.release()


    def get_transcription(self,file,language) -> str:
        """
        Generate a transcription
        
        file (file): Audio file to trancribe
        language: Language of the text

        Returns:
            str: Transcription text
        """
        # Create the initial instruction
        instructions = [int(self.configuration["DEFAULT"]["START_TOKEN_ID"])]

        # Check if language provided to add to the instrctions
        #TODO
        #if language:
        #    instructions.append("hola")

        # Read the bytes from the audio file
        transcription = self.transcribe(instructions, file)

        # Return the transcription
        return transcription