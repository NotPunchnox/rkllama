import logging
import os

logger = logging.getLogger("rkllama.stt")

# SUPPORTED STT MODELS
WHISPER = "whisper.ini"
OMNI_ASR = "omniasr.txt"
    
def generate_transcription(model_path,file,language) -> str:
    """
    Generate a transcription
    
    model_path (str): Path of the stt model
    file (file): Audio file to trancribe
    language: Language of the text

    Returns:
        str: Transcription text
    """

    # CHeck the model type
    model_type = check_stt_model_type(model_path)    
    logger.debug(f"Detected STT model = {model_type}")

    # Depending of the model type, import the correct logic
    if model_type == WHISPER:
        # It is whisper model call whisper logic
        from .models.audio.whisper import WhisperSTTModelRKNN
        model = WhisperSTTModelRKNN(model_path)
    else:
        # Default OmniASR. Call this logic
        from .models.audio.omniasr import OmniCtcSTTModelRKNN
        model = OmniCtcSTTModelRKNN(model_path)

    # Generate the transcription
    transcription = model.get_transcription(file,language)
    logger.debug(f"Returned transcription = {transcription}")

    # Return the transcription
    return transcription


def generate_translation(model_path,file,language) -> str:
    """
    Generate a translation
    
    model_path (str): Path of the stt model
    file (file): Audio file to translate
    language: Language of the text

    Returns:
        str: Translation text
    """

    # CHeck the model type
    model_type = check_stt_model_type(model_path)    
    logger.debug(f"Detected STT model = {model_type}")

    # Depending of the model type, import the correct logic
    if model_type == WHISPER:
        # It is whisper model call whisper logic
        from .models.audio.whisper import WhisperSTTModelRKNN
        model = WhisperSTTModelRKNN(model_path)

    # Generate the translation
    translation = model.get_translation(file,language)
    logger.debug(f"Returned translation = {translation}")

    # Return the translation
    return translation


def check_stt_model_type(model_path: str) -> str:
    """
    Check the STT model type.

    Returns:
        Type of the STT model

    """
    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"Not a model directory: {model_path}")

    if os.path.isfile(os.path.join(model_path, WHISPER)):
        # It is a STT whisper model
        return WHISPER
    else:    
        # Default omniASR
        return OMNI_ASR