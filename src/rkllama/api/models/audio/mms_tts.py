import numpy as np
import os
import soundfile as sf
from typing import Tuple
from rknnlite.api.rknn_lite import RKNNLite
import configparser
import torch
import torch.nn as nn
import json
import io
from pydub import AudioSegment
import uuid
import logging
import time


logger = logging.getLogger("rkllama.audio.mms_tts")

class MMSTTSModelRKNN:
    def __init__(
        self,
        model_path: str,
    ):

        # Get encoder, decoder and configuration 
        encoder, decoder, vocab = self.find_model_files(model_path)

        # Prepare the RKNN runtime model
        # Encoder
        self.encoder_rknn = RKNNLite(verbose=False)
        self.encoder_rknn.load_rknn(encoder)
        self.encoder_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        # Decoder
        self.decoder_rknn = RKNNLite(verbose=False)
        self.decoder_rknn.load_rknn(decoder)
        self.decoder_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

        # Save the vocab of the  model
        self.vocab= self.read_json_file(vocab)

        # Save the model directory
        self.model_path = model_path

        # Load Inputs and Outputs of the models
        self.load_inputs_outputs_from_rknn_models()
    


    def read_json_file(self, file_path):
        """
        Reads and parses a JSON file safely.
        
        :param file_path: Path to the JSON file
        :return: Parsed Python object (dict, list, etc.) or None if error
        """
        # Check if file exists
        if not os.path.isfile(file_path):
            print(f"Error: File '{file_path}' not found.")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # Parse JSON into Python object
                return data
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format. Details: {e}")
        except Exception as e:
            print(f"Error reading file: {e}")
        
        return None


    def find_model_files(self, dir_path: str) -> Tuple[str, str, str]:
        """
        Find exactly one .rknn for encoder, one .rknn for decoder and a vocab file for mms_tts model.

        Returns:
            (rknn_path, rknn_path, configurvocabation file path)

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

        # Search for the vocab of the model
        vocab = os.path.join(dir_path, "mms_tts.json")

        # Return the files    
        return ( encoder_rknn, decoder_rknn, vocab)
    
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

        # Get the inputs and outputs name in the order expected by the model
        self.decoder_rknn_order_inputs_names = [ decoder_runtime.get_tensor_attr(i, is_output=False).name.decode("utf-8") for i in range(n_input_decoder) ]
        self.decoder_rknn_order_output_names = [ decoder_runtime.get_tensor_attr(i, is_output=True).name.decode("utf-8") for i in range(n_output_decoder) ]

        # Get the max lenght allowed by the encoder and decoder
        self.max_length = encoder_runtime.get_tensor_attr(0, is_output=False).dims[1]


    def release_rknn_models(self):
        # Release resources from RKNN
        self.encoder_rknn.release()
        self.decoder_rknn.release()
        
    def run_encoder(self,input_ids_array, attention_mask_array):
        log_duration, input_padding_mask, prior_means, prior_log_variances = self.encoder_rknn.inference(inputs=[input_ids_array, attention_mask_array])
        
        return log_duration, input_padding_mask, prior_means, prior_log_variances

    def run_decoder(self, attn, output_padding_mask, prior_means, prior_log_variances):
        waveform  = self.decoder_rknn.inference(inputs=[attn, output_padding_mask, prior_means, prior_log_variances])[0]
        
        return waveform

    def pad_or_trim(self, token_id, attention_mask):
        pad_len = self.max_length - len(token_id)
        if pad_len <= 0:
            token_id = token_id[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        if pad_len > 0:
            token_id = token_id + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return token_id, attention_mask

    def preprocess_input(self, text):
        text = list(text.lower())
        input_id = []
        for token in text:
            if token not in self.vocab:
                continue
            input_id.append(0)
            input_id.append(int(self.vocab[token]))
        input_id.append(0)
        attention_mask = [1] * len(input_id)

        input_id, attention_mask = self.pad_or_trim(input_id, attention_mask)

        input_ids_array = np.array(input_id)[None,...]
        attention_mask_array = np.array(attention_mask)[None,...]

        return input_ids_array, attention_mask_array

    def middle_process(self, log_duration, input_padding_mask):
        log_duration = torch.tensor(log_duration)
        input_padding_mask = torch.tensor(input_padding_mask)
        
        speaking_rate = 1
        length_scale = 1.0 / speaking_rate
        duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

        # Create a padding mask for the output lengths of shape (batch, 1, max_output_length)
        predicted_lengths_max_real = predicted_lengths.max()
        predicted_lengths_max = self.max_length * 2

        indices = torch.arange(predicted_lengths_max, dtype=predicted_lengths.dtype)
        output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
        output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

        # Reconstruct an attention tensor of shape (batch, 1, out_length, in_length)
        attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
        batch_size, _, output_length, input_length = attn_mask.shape
        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        indices = torch.arange(output_length, dtype=duration.dtype)
        valid_indices = indices.unsqueeze(0) < cum_duration
        valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
        padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
        attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

        attn = attn.numpy()
        output_padding_mask = output_padding_mask.numpy()
        
        return attn, output_padding_mask, predicted_lengths_max_real

    def generate_speech(self, input,voice = None,response_format = None,stream_format = None,speed = None) -> tuple[bytes, str]:
    
        # Prepare input for the model
        input_ids_array, attention_mask_array = self.preprocess_input(input)

        # Encode
        encoder_start = time.time()
        log_duration, input_padding_mask, prior_means, prior_log_variances = self.run_encoder(input_ids_array, attention_mask_array)
        logger.debug(f"Encoder time: {(time.time() - encoder_start):.2f}s")

        # Middle process
        attn, output_padding_mask, predicted_lengths_max_real = self.middle_process(log_duration, input_padding_mask)

        # Decode
        decoder_start = time.time()
        waveform = self.run_decoder(attn, output_padding_mask, prior_means, prior_log_variances)
        logger.debug(f"Decoder time: {(time.time() - decoder_start):.2f}s")

        # Generate a temp output wav file
        temp_output_path = f"{self.model_path}/{uuid.uuid4()}.wav"
        sf.write(file=temp_output_path, data=np.array(waveform[0][:predicted_lengths_max_real * 256]), samplerate=16000)

        # Return the audio in bytes format
        return convert_wav_to_bytes(temp_output_path, "wav") 


def convert_wav_to_bytes(wav_path: str, output_format: str) -> tuple[bytes, str]:
    """
    Returns:
        wav_bytes     : Original WAV buffer
        media_type    : MIME type of converted audio
        audio_bytes   : Converted audio buffer
    """

    # Check supported formats
    output_format = output_format.lower()
    supported = {"mp3", "opus", "aac", "flac", "pcm", "wav"}
    if output_format not in supported:
        raise ValueError(f"Unsupported format: {output_format}")

    # Load WAV
    audio = AudioSegment.from_wav(wav_path)
    
    # Delete file from filesystem. ALready loaded in memory AudioSegment
    os.remove(wav_path)

    # Out buffer bytes
    out_buffer = io.BytesIO()

    if output_format == "wav":
        audio.export(out_buffer, format="wav")
        media_type = "audio/wav"

    elif output_format == "mp3":
        audio.export(out_buffer, format="mp3")
        media_type = "audio/mpeg"

    elif output_format == "opus":
        audio.export(out_buffer, format="opus")
        media_type = "audio/opus"

    elif output_format == "aac":
        audio.export(out_buffer, format="adts")
        media_type = "audio/aac"

    elif output_format == "flac":
        audio.export(out_buffer, format="flac")
        media_type = "audio/flac"

    elif output_format == "pcm":
        out_buffer.write(audio.raw_data)
        media_type = "audio/pcm"

    # Return the audio bytes
    return out_buffer.getvalue(), media_type
