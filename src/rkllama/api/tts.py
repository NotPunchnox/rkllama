import json
import os
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Sequence
import wave
import numpy as np
import onnxruntime
import io
from pydub import AudioSegment
import uuid
from piper.tashkeel import TashkeelDiacritizer
from piper.phonemize_espeak import ESPEAK_DATA_DIR
from piper import PiperVoice, SynthesisConfig, PiperConfig
from dataclasses import dataclass

from rknnlite.api import RKNNLite

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)

_DEFAULT_SYNTHESIS_CONFIG = SynthesisConfig()

_LOGGER = logging.getLogger(__name__)

@dataclass
class PhonemeAlignment:
    phoneme: str
    phoneme_ids: Sequence[int]
    num_samples: int

class PiperVoiceRKNN(PiperVoice):
    """A voice for Piper."""

    session: onnxruntime.InferenceSession
    """ONNX session."""

    session_rknn = RKNNLite
    """RKNN session."""
    
    config: PiperConfig
    """Piper voice configuration."""

    espeak_data_dir: Path = ESPEAK_DATA_DIR
    """Path to espeak-ng data directory."""

    # For Arabic text only
    use_tashkeel: bool = True
    tashkeel_diacritizier: Optional[TashkeelDiacritizer] = None
    taskeen_threshold: Optional[float] = 0.8

    def __init__(self, *args, session_rknn, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_rknn = session_rknn

    @staticmethod
    def load(
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        use_cuda: bool = False,
        espeak_data_dir: Union[str, Path] = ESPEAK_DATA_DIR,
    ) -> "PiperVoiceRKNN":
        """
        Load an ONNX model and config.

        :param model_path: Path to ONNX voice model.
        :param config_path: Path to JSON voice config (defaults to model_path + ".json").
        :param use_cuda: True if CUDA (GPU) should be used instead of CPU.
        :param espeak_data_dir: Path to espeak-ng data dir (defaults to internal data).
        :return: Voice object.
        """

        if config_path is None:
            config_path = f"{model_path}.json"
            _LOGGER.debug("Guessing voice config path: %s", config_path)

        providers: list[Union[str, tuple[str, dict[str, Any]]]]
        if use_cuda:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_algo_search": "HEURISTIC"},
                )
            ]
            _LOGGER.debug("Using CUDA")
        else:
            providers = ["CPUExecutionProvider"]


        # Get encoder, decoder and config file from models path
        encoder, decoder, config_path = find_model_files(model_path)

        # Read the config file
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)

        # Load RKNN
        decoder_rknn = RKNNLite(verbose=False)
        decoder_rknn.load_rknn(decoder)
        
        return PiperVoiceRKNN(
            config=PiperConfig.from_dict(config_dict),
            session=onnxruntime.InferenceSession(
                str(encoder),
                sess_options=onnxruntime.SessionOptions(),
                providers=providers,
            ),
            session_rknn= decoder_rknn,
            espeak_data_dir=Path(espeak_data_dir),
        )

    def synthesize_wav(
        self,
        text: str,
        wav_file: wave.Wave_write,
        syn_config: Optional[SynthesisConfig] = None,
        set_wav_format: bool = True,
        include_alignments: bool = False,
    ) -> Optional[list[PhonemeAlignment]]:
        """
        Synthesize and write WAV audio from text.

        :param text: Text to synthesize.
        :param wav_file: WAV file writer.
        :param syn_config: Synthesis configuration.
        :param set_wav_format: True if the WAV format should be set automatically.
        :param include_alignments: If True and the model supports it, return phoneme/audio alignments.

        :return: Phoneme/audio alignments if include_alignments is True, otherwise None.
        """

        # Init runtime RKNN for inference
        self.session_rknn.init_runtime()

        alignments: list[PhonemeAlignment] = []
        first_chunk = True
        for audio_chunk in self.synthesize(
            text, syn_config=syn_config
        ):
            if first_chunk:
                if set_wav_format:
                    # Set audio format on first chunk
                    wav_file.setframerate(audio_chunk.sample_rate)
                    wav_file.setsampwidth(audio_chunk.sample_width)
                    wav_file.setnchannels(audio_chunk.sample_channels)

                first_chunk = False

            wav_file.writeframes(audio_chunk.audio_int16_bytes)

            if include_alignments and audio_chunk.phoneme_alignments:
                alignments.extend(audio_chunk.phoneme_alignments)
        
        # Release RKNNLite resources before return
        self.session_rknn.release()
        
        if include_alignments:
            return alignments

        return None
    
    def phoneme_ids_to_audio(
        self,
        phoneme_ids: list[int],
        syn_config: Optional[SynthesisConfig] = None,
        include_alignments: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Synthesize raw audio from phoneme ids.

        :param phoneme_ids: List of phoneme ids.
        :param syn_config: Synthesis configuration.
        :param include_alignments: Return samples per phoneme id if True.
        :return: Audio float numpy array from voice model (unnormalized, in range [-1, 1]).

        If include_alignments is True and the voice model supports it, the return
        value will be a tuple instead with (audio, phoneme_id_samples) where
        phoneme_id_samples contains the number of audio samples per phoneme id.
        """

        if syn_config is None:
            syn_config = _DEFAULT_SYNTHESIS_CONFIG

        speaker_id = syn_config.speaker_id
        length_scale = syn_config.length_scale
        noise_scale = syn_config.noise_scale
        noise_w_scale = syn_config.noise_w_scale

        if length_scale is None:
            length_scale = self.config.length_scale

        if noise_scale is None:
            noise_scale = self.config.noise_scale

        if noise_w_scale is None:
            noise_w_scale = self.config.noise_w_scale

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_w_scale],
            dtype=np.float32,
        )

        args = {
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths,
            "scales": scales,
        }

        if self.config.num_speakers <= 1:
            speaker_id = None

        if (self.config.num_speakers > 1) and (speaker_id is None):
            # Default speaker
            speaker_id = 0

        if speaker_id is not None:
            sid = np.array([speaker_id], dtype=np.int64)
            args["sid"] = sid

        # Encoder through onnx
        encoder_output = self.session.run(
            None,
            args,
        )
        
        # Get the encoder outputs
        if speaker_id is not None:
            z, y_mask, _ = encoder_output
        else:
            z, y_mask = encoder_output

        
        # Get the input time expected by the RKNN model for chunk processing
        static_input_value_rknn = self.session_rknn.rknn_runtime.get_tensor_attr(0).dims[2]
   

        # Chunk z and y_mask into smaller pieces
        z_chunks, _ = self.chunk_tensor(z, chunk_size=static_input_value_rknn, overlap=0)
        y_chunks, _ = self.chunk_tensor(y_mask, chunk_size=static_input_value_rknn, overlap=0)

        # Process each chunk through RKNN decoder
        audio_chunks = []
        for zc, yc in zip(z_chunks, y_chunks):

            # Get the current input time of the chunk
            input_value_chunk = zc.shape[2]
            
            # Pad zc and y_maskc if needed for the expected input size of RKNN
            if input_value_chunk < static_input_value_rknn:
                
                padded_z = np.zeros((zc.shape[0], zc.shape[1], static_input_value_rknn), dtype=np.float32)
                padded_z[:, :, :input_value_chunk] = zc
                zc = padded_z

                padded_mask = np.zeros((yc.shape[0], yc.shape[1], static_input_value_rknn), dtype=np.float32)
                padded_mask[:, :, :input_value_chunk] = yc
                yc = padded_mask

            # Construct inputs for RKNN decoder model
            inputs_chunk = [zc.astype(np.float32), yc.astype(np.float32)]

            # Inference RKNN (decoder) of the chunk
            result = self.session_rknn.inference(inputs=inputs_chunk, data_format="nchw")

            # Check if current tensor was padded to remove the generated junk final audio by the decoder
            if input_value_chunk < static_input_value_rknn:
                result = self.fix_rknn_output(result, input_value_chunk, static_input_value_rknn)        

            # Add the output of the current chunk decoder to the list
            audio_chunks.append(result)    
        
        # Concatenate all audio chunks        
        audio = np.concatenate(audio_chunks, axis=-1)
        
        # Continue original Piper logic
        audio = audio.squeeze()
        if not include_alignments:
            return audio

        if len(result) == 1:
            # Alignment is not available from voice model
            return audio, None

        # Number of samples for each phoneme id
        phoneme_id_samples = (result[1].squeeze() * self.config.hop_length).astype(
            np.int64
        )

        return audio, phoneme_id_samples


    def fix_rknn_output(self, result, sz, static_t):
        """
        Remove the generated silence sound because the final padding zeros in the input of the decoder 
        """
        # Generated Shape by the decoder
        tensor = result[0]
        orig_shape = tensor.shape

        # Reshape with the correct size without the zeros (blank audio)
        flat = tensor.reshape(-1)
        real_len = int(flat.shape[0] * sz / static_t)
        flat = flat[:real_len]

        # Replace the orginal tensor with the fixed one
        new_shape = list(orig_shape)
        new_shape[-1] = real_len

        # Return the fixed tensor
        return [flat.reshape(new_shape)]


    def chunk_tensor(self, t, chunk_size=55, overlap=10):
        """
        Split a tensor (1, C, T) in chunks with overlapping.
        
        - t: numpy array shape (1, C, T)
        - chunk_size: fixed size required by decoder RKNN
        - overlap: Overlap between chunks
        """
        assert t.ndim == 3, "Tensor must be (1, C, T)"

        # Define the steps
        _, C, T = t.shape
        step = chunk_size - overlap

        chunks = []
        ranges = []
        # Loop over the tensor to chunk
        start = 0
        while start < T:
            end = min(start + chunk_size, T)
            chunk = t[:, :, start:end]

            chunks.append(chunk)
            ranges.append((start, end))

            start += step

        # return the chunks and the range between them
        return chunks, ranges


def generate_speech(model_piper_path,input,voice,response_format,stream_format,volume,length_scale,noise_scale,noise_w_scale,normalize_audio) -> tuple[bytes, str]:
    """
    Returns:
        Return the bytes from a generated speech
    """

    # Load the Piper instance for RKNN
    piperTTS = PiperVoiceRKNN.load(model_path=model_piper_path)
    
    # Generate a temp output wav file
    temp_output_path = f"{model_piper_path}/{uuid.uuid4()}.wav"

    # Create the empyty out file and call piper
    with wave.open(temp_output_path, "wb") as wav_file:

        # Create empty config. By default the .sjon is used
        syn_config = SynthesisConfig()

        # Update the config if indicated in the request
        if volume:
            syn_config.volume = volume
        if length_scale:
            syn_config.length_scale = length_scale
        if noise_scale:
            syn_config.noise_scale = noise_scale
        if noise_w_scale:
            syn_config.noise_w_scale = noise_w_scale
        if normalize_audio:
            syn_config.normalize_audio = normalize_audio
        if voice and voice in piperTTS.config.speaker_id_map:
            syn_config.speaker_id = piperTTS.config.speaker_id_map.get(voice)    

        # Generate the speech    
        piperTTS.synthesize_wav(input, wav_file, syn_config=syn_config)

        # Return the audio in bytes format
        return convert_wav_to_bytes(temp_output_path, response_format) 
        

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


def find_model_files(dir_path: str) -> Tuple[str, str, str]:
    """
    Find exactly one .onnx, .rknn and .json file in a piper model directory.

    Returns:
        (onnx_path, rknn_path, json_path)

    Raises:
        FileNotFoundError: if a required file type is missing
        ValueError: if more than one file of a type is found
    """

    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    encoder_onnx = None
    decoder_rknn = None
    config_json = None

    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if not os.path.isfile(path):
            # Skip dictionaries
            continue

        # Check the current file in the directoty loop
        ext = os.path.splitext(name)[1].lower()
        if ext == ".onnx":
            if not encoder_onnx:
                encoder_onnx = path
        elif ext == ".rknn":
            if not decoder_rknn:
                decoder_rknn = path
        elif ext == ".json":
            if not config_json:
                config_json = path
        
        if encoder_onnx and decoder_rknn and config_json:
            # All files found. Exit loop
            break

    # Return the files    
    return ( encoder_onnx, decoder_rknn, config_json)
