# import gradio as gr
# import styletts2importable
# import os
# import numpy as np
# import tempfile
# from txtsplit import txtsplit

# # Other imports and initialization remain the same...
# voicelist = ['f-us-1', 'f-us-2', 'f-us-3', 'f-us-4', 'm-us-1', 'm-us-2', 'm-us-3', 'm-us-4']
# voices = {}

# def synthesize_and_save(text, voice, multispeakersteps):
#     if text.strip() == "":
#         raise gr.Error("You must enter some text")
#     if len(text) > 50000:
#         raise gr.Error("Text must be <50k characters")

#     texts = txtsplit(text)
#     v = voice.lower()
#     audios = []

#     for t in texts:
#         audios.append(styletts2importable.inference(t, voices[v], alpha=0.3, beta=0.7, diffusion_steps=multispeakersteps, embedding_scale=1))
    
#     # Concatenate the audio clips into one
#     final_audio = np.concatenate(audios)
    
#     # Save the audio to a temporary file
#     sample_rate = 24000
#     temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#     temp_file_path = temp_file.name
    
#     # Use a library to save the concatenated audio, e.g., `scipy.io.wavfile` or similar
#     from scipy.io.wavfile import write
#     write(temp_file_path, sample_rate, final_audio)
    
#     return temp_file_path, (sample_rate, final_audio)

# with gr.Blocks() as vctk:
#     with gr.Row():
#         with gr.Column(scale=1):
#             inp = gr.Textbox(label="Text", info="What would you like StyleTTS 2 to read? It works better on full sentences.", interactive=True)
#             voice = gr.Dropdown(voicelist, label="Voice", info="Select a default voice.", value='m-us-2', interactive=True)
#             multispeakersteps = gr.Slider(minimum=3, maximum=15, value=3, step=1, label="Diffusion Steps", info="Theoretically, higher should be better quality but slower. Try with lower steps first.", interactive=True)
        
#         with gr.Column(scale=1):
#             btn = gr.Button("Synthesize", variant="primary")
#             audio = gr.Audio(interactive=False, label="Synthesized Audio", waveform_options={'waveform_progress_color': '#3C82F6'})
#             download_btn = gr.File(label="Download Synthesized Audio")
            
#             # When button is clicked, synthesize the audio and save it, then provide a download link
#             btn.click(synthesize_and_save, inputs=[inp, voice, multispeakersteps], outputs=[download_btn, audio], concurrency_limit=4)

# if __name__ == "__main__":
#     demo.queue(api_open=False, max_size=15).launch(show_api=False)
