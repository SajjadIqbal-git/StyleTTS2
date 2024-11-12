from fastapi import FastAPI, HTTPException,UploadFile, File,Form
from pydantic import BaseModel
import styletts2importable
import ljspeechimportable
import torch
import numpy as np
from txtsplit import txtsplit
import base64
from fastapi.responses import FileResponse
from pydub import AudioSegment
import tempfile
from tempfile import NamedTemporaryFile
from scipy.io.wavfile import write  
import os
from fastapi.responses import JSONResponse

app = FastAPI()

voicelist = ['f-us-1', 'f-us-2', 'f-us-3', 'f-us-4', 'm-us-1', 'm-us-2', 'm-us-3', 'm-us-4']
voices = {}

# Pre-compute voice styles
for v in voicelist:
    voices[v] = styletts2importable.compute_style(f'voices/{v}.wav')

class SynthesizeRequest(BaseModel):
    text: str
    voice: str
    lngsteps: int


class CLSynthesizeRequest(BaseModel):
    text: str
    vcsteps: int
    embscale: float
    alpha: float
    beta: float


class LJSynthesizeRequest(BaseModel):
    text: str
    steps: int


@app.post("/synthesize/")
async def synthesize(request: SynthesizeRequest):
    text = request.text.strip()
    voice = request.voice.lower()
    lngsteps = request.lngsteps
    
    if not text:
        raise HTTPException(status_code=400, detail="You must enter some text")
    if len(text) > 50000:
        raise HTTPException(status_code=400, detail="Text must be <50k characters")

    print("*** saying ***")
    print(text)
    print("*** end ***")

    texts = txtsplit(text)
    audio_segments = []

    for t in texts:
        audio_data = styletts2importable.inference(t, voices[voice], alpha=0.3, beta=0.7, diffusion_steps=lngsteps, embedding_scale=1)
        
        # Convert audio_data (numpy array) to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_file_path = wav_file.name
            audio_data = (audio_data * 32767).astype(np.int16)  # Scale to int16
            write(wav_file_path, 24000, audio_data)

        # Convert WAV to MP3 using pydub
        mp3_file_path = wav_file_path.replace(".wav", ".mp3")
        AudioSegment.from_wav(wav_file_path).export(mp3_file_path, format="mp3")
        
        audio_segments.append(mp3_file_path)

    # Return the first audio segment for simplicity
    return FileResponse(audio_segments[0], media_type="audio/mpeg", filename="synthesized_audio.mp3")





@app.post("/clsynthesize/")
async def clsynthesize(
    text: str = Form(...),
    vcsteps: int = Form(...),
    embscale: float = Form(...),
    alpha: float = Form(...),
    beta: float = Form(...),
    voice: UploadFile = File(...)
):
    text = text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="You must enter some text")
    if len(text) > 50000:
        raise HTTPException(status_code=400, detail="Text must be <50k characters")
    if embscale > 1.3 and len(text) < 20:
        return {"warning": "WARNING: Short text may result in static."}

    # Read the uploaded voice file
    voice_audio = await voice.read()

    # Save the uploaded voice file temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_voice_file:
        temp_voice_path = temp_voice_file.name
        temp_voice_file.write(voice_audio)

    try:
        # Compute the style for the provided voice audio file
        voice_style = styletts2importable.compute_style(temp_voice_path)
    except Exception as e:
        os.remove(temp_voice_path)
        raise HTTPException(status_code=500, detail=f"Error in computing style: {str(e)}")

    texts = txtsplit(text)  # Split text into smaller segments for processing
    combined_audio = []

    for t in texts:
        try:
            audio_data = styletts2importable.inference(
                t, voice_style, alpha=alpha, beta=beta, diffusion_steps=vcsteps, embedding_scale=embscale
            )
            audio_data = (audio_data * 32767).astype(np.int16)  # Convert to int16 format
            combined_audio.extend(audio_data)
        except Exception as e:
            os.remove(temp_voice_path)
            raise HTTPException(status_code=500, detail=f"Error in inference: {str(e)}")

    # Save the combined audio as a WAV file first
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = wav_file.name
        write(wav_path, 24000, np.array(combined_audio, dtype=np.int16))

    # Convert WAV to MP3
    mp3_path = wav_path.replace(".wav", ".mp3")
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")

    # Remove the temporary WAV file
    os.remove(wav_path)
    os.remove(temp_voice_path)

    # Return the MP3 file as a downloadable link
    return FileResponse(mp3_path, media_type="audio/mpeg", filename="output.mp3")


@app.post("/ljsynthesize/")
async def ljsynthesize(request: LJSynthesizeRequest):
    text = request.text.strip()
    steps = request.steps
    
    if not text:
        raise HTTPException(status_code=400, detail="You must enter some text")
    if len(text) > 150000:
        raise HTTPException(status_code=400, detail="Text must be <150k characters")
    
    print("*** saying ***")
    print(text)
    print("*** end ***")

    texts = txtsplit(text)
    audios = []
    noise = torch.randn(1, 1, 256).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    for t in texts:
        audio_data = ljspeechimportable.inference(t, noise, diffusion_steps=steps, embedding_scale=1)
        # Convert the audio data to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        audios.append(audio_base64)
    
    return {"sample_rate": 24000, "audio": audios}



@app.get("/")
def read_root():
    return {"message": "Welcome to the StyleTTS 2 API. Use /synthesize, /clsynthesize, or /ljsynthesize to generate audio."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
