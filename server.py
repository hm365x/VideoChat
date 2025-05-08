import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModel, AutoTokenizer
import torch
import uvicorn
from transformers.generation.streamers import BaseStreamer
from threading import Thread
import queue

from src.thg import Muse_Talk


# GLM-4-Voice
class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt

        # variables used in the streaming process
        self.token_queue = queue.Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class MllmWorker:
    def __init__(self, model_path = "./weights/ZhipuAI/glm-4-voice-9b", device='cuda'):
        self.device = device
        self.glm_model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                                   device=device).to(device).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.glm_tokenizer, self.glm_model

        prompt = params["prompt"]

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = int(params.get("max_new_tokens", 256))

        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        streamer = TokenStreamer(skip_prompt=True)
        thread = Thread(target=model.generate,
                        kwargs=dict(**inputs, max_new_tokens=int(max_new_tokens),
                                    temperature=float(temperature), top_p=float(top_p),
                                    streamer=streamer))
        thread.start()
        for token_id in streamer:
            yield (json.dumps({"token_id": token_id, "error_code": 0}) + "\n").encode()

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": "Server Error",
                "error_code": 1,
            }
            yield (json.dumps(ret)+ "\n").encode()


# MuseTalk
# class ThgWorker:
#     def init(self):
#         self.thg = Muse_Talk()
#         self.thg.warm_up()
    
#     def generate_frames(self, project_path, audio_path, avatar_name):
#         self.thg.infer(project_path=project_path, audio_path=llm_response_audio, avatar_name=avatar_name)


# FastAPI
app = FastAPI()


@app.post("/generate_stream")
async def generate_stream(request: Request):
    params = await request.json()

    generator = mllm_worker.generate_stream_gate(params)
    return StreamingResponse(generator)


if __name__ == "__main__":
    # glm-4-voice worker
    mllm_worker = MllmWorker()

    # thg worker
    # thg_worker = ThgWorker()

    #uvicorn.run(app, host="localhost", port=10000, log_level="info") # localhost有可能解析为 127.0.0.1 或 ::1, 后者可能绑定失败
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="info")
