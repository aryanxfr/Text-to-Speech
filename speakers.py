from TTS.api import TTS
model_name = TTS.list_models()[0]
tts = TTS(model_name)
print(f"Available speakers: {tts.speakers}")