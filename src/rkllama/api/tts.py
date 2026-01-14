import logging
import os

logger = logging.getLogger("rkllama.tts")

# SUPPORTED TTS MODELS
PIPER = "piper.json"
MMS_TTS = "mms_tts.json"
    

def generate_speech(model_path,input,voice,response_format,stream_format,speed) -> bytes:
    """
    Returns:
        Return the bytes from a generated speech
    """

    # CHeck the model type
    model_type = check_tts_model_type(model_path)    
    logger.debug(f"Detected TTS model = {model_type}")

    # Depending of the model type, import the correct logic
    if model_type == PIPER:
        # It is piper model call piper logic
        from .models.audio.piper import PiperVoiceRKNN
        model = PiperVoiceRKNN.load(model_path= model_path)
    elif model_type == MMS_TTS:   
        # It is mms model call mms logic
        from .models.audio.mms_tts import MMSTTSModelRKNN
        model = MMSTTSModelRKNN(model_path=model_path)
   
    # Generate the speech
    logger.debug(f"Generating speech for model {model_type} for text = {input}")
    speech = model.generate_speech(input,voice,response_format,stream_format,speed)

    # Release RKNN resources
    model.release_rknn_models()

    # Return the speech
    return speech


def check_tts_model_type(model_path: str) -> str:
    """
    Check the TTS model type.

    Returns:
        Type of the TTS model

    """
    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"Not a model directory: {model_path}")

    if os.path.isfile(os.path.join(model_path, MMS_TTS)):
        # It is a STT whisper model
        return MMS_TTS
    else:    
        # Default PIPER
        return PIPER
    