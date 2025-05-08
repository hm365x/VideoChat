import json
import os.path
import tempfile
import sys
import re
import uuid
import requests
import gradio as gr
import torch
import torchaudio
import time
import queue
from transformers import WhisperFeatureExtractor, AutoTokenizer, AutoModel

from src.GLM_4_Voice.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from src.GLM_4_Voice.speech_tokenizer.utils import extract_speech_token
from src.GLM_4_Voice.flow_inference import AudioDecoder


# Regular expression pattern for audio tokens
audio_token_pattern = re.compile(r"<\|audio_(\d+)\|>")


@torch.no_grad()
class GLM_4_Voice:
    def __init__(self):
        self.flow_path = "./weights/ZhipuAI/glm-4-voice-decoder"
        self.model_path = "./weights/ZhipuAI/glm-4-voice-9b"
        self.tokenizer_path = "./weights/ZhipuAI/glm-4-voice-tokenizer"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.glm_tokenizer = None
        self.audio_decoder = None
        self.whisper_model = None
        self.feature_extractor = None
        self.load_weights()
        # self.warm_up()

    def load_weights(self):
        if self.audio_decoder is not None:
            return

        # GLM
        self.glm_tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Flow & Hift
        flow_config = os.path.join(self.flow_path, "config.yaml")
        flow_checkpoint = os.path.join(self.flow_path, 'flow.pt')
        hift_checkpoint = os.path.join(self.flow_path, 'hift.pt')
        self.audio_decoder = AudioDecoder(
            config_path=flow_config, 
            flow_ckpt_path=flow_checkpoint,
            hift_ckpt_path=hift_checkpoint,
            device=self.device
            )

        # Speech tokenizer
        self.whisper_model = WhisperVQEncoder.from_pretrained(self.tokenizer_path).eval().to(self.device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.tokenizer_path)


    def warm_up(self):
        class User_Input:
            def __init__(self, user_input):
                self.text = user_input
                self.files = None

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.infer(tmpdirname, User_Input('你好'), "", queue.Queue())
       

    def infer(
        self,
        project_path,
        user_input,
        user_messages, 
        mllm_queue,
        temperature = 0.2,
        top_p = 0.8,
        max_new_token = 1000,
    ):
        try:
            audio_path = f"{project_path}/audio"
            os.makedirs(audio_path, exist_ok=True)

            # Extract tokens 
            if user_input.files:
                input_audio_path = user_input.files[0].path
                audio_tokens = extract_speech_token(
                    self.whisper_model, self.feature_extractor, [input_audio_path]
                )[0]
                if len(audio_tokens) == 0:
                    raise gr.Error("No audio tokens extracted")
                audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
                audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
                user_input_tokens = audio_tokens
                system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

            else:
                input_text = user_input.text
                if input_text == "":
                    raise gr.Error("Please provide a non-empty input")
                user_input_tokens = input_text
                system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."


            # Gather history
            input_tokens = user_messages.strip()
            if "<|system|>" not in input_tokens:
                input_tokens += f"<|system|>\n{system_prompt}"
            input_tokens += f"<|user|>\n{user_input_tokens}<|assistant|>streaming_transcription\n"

            # Generate results
            response = requests.post(
                "http://localhost:10000/generate_stream",
                data=json.dumps({
                    "prompt": input_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_token,
                }),
                stream=True
            )
            text_tokens, audio_tokens = [], []
            audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
            end_token_id = self.glm_tokenizer.convert_tokens_to_ids('<|user|>')

            complete_tokens = []
            complete_text = ""
            prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(self.device)
            this_uuid = str(uuid.uuid4())
            tts_speechs = []
            tts_mels = []
            prev_mel = None
            is_finalize = False
            block_size = 15
            index = 0

            start_time = time.time()
            for chunk in response.iter_lines():
                token_id = json.loads(chunk)["token_id"]
                if token_id == end_token_id:
                    is_finalize = True
                if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                    block_size = 30
                    tts_token = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)

                    if prev_mel is not None:
                        prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                    tts_speech, tts_mel = self.audio_decoder.token2wav(tts_token, uuid=this_uuid,
                                                                prompt_token=flow_prompt_speech_token.to(self.device),
                                                                prompt_feat=prompt_speech_feat.to(self.device),
                                                                finalize=is_finalize)
                    prev_mel = tts_mel
                    tts_speechs.append(tts_speech.squeeze())
                    tts_mels.append(tts_mel)
    
                    # Save audio to wav and text
                    output_wav_path = f"{audio_path}/llm_response_audio_{index}.wav"
                    merge_time = time.time()
                    torchaudio.save(output_wav_path, tts_speech.cpu(), 22050, format="wav")

                    text = self.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)
                    complete_text += text
                    text_tokens = []

                    mllm_queue.put((text, output_wav_path))

                    audio_seconds = torchaudio.info(output_wav_path).num_frames / 22050
                    print(f"Clip {index} length:{audio_seconds:.2f} s, infer cost {time.time()-start_time}, text : {text}")
                    index += 1
                    start_time = time.time()

                    flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                    audio_tokens = []

                if not is_finalize:
                    complete_tokens.append(token_id)
                    if token_id >= audio_offset:
                        audio_tokens.append(token_id - audio_offset)
                    else:
                        text_tokens.append(token_id)
            
            mllm_queue.put((None, None))
            user_messages = input_tokens + complete_text

            return user_messages


        except Exception as e:
            print(e)
            raise gr.Error("An error occurred during inference")
