from openai import OpenAI
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer, TextIteratorStreamer
from threading import Thread
import re
import queue
import os
import json
import time

if os.path.exists("/root/autodl-tmp"):
    custom_cache_dir = "/root/autodl-tmp/llmcache"
elif os.path.exists("/mnt/d/wsl/autodl-tmp"):
    custom_cache_dir = "/mnt/d/wsl/autodl-tmp/llmcache"
else:
    print(f"llm using cache directory: ~/.cache")
if custom_cache_dir:
    print(f"llm using cache directory: {custom_cache_dir}")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(custom_cache_dir, "huggingface")
    os.environ["MODELSCOPE_CACHE"] = os.path.join(custom_cache_dir, "modelscope")
#给 Qwen\__init__\from_pretrained 下载模型指定 cache_dir 位置，否则默认会将模型缓存到系统盘 ~/.cache（/root/.cache）
# Qwen2.5系列: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
# Qwen/Qwen2.5-0.5B-Instruct, 9.1G显存
# Qwen/Qwen2.5-1.5B-Instruct, 11.3G爆12G显存
# Qwen/Qwen2.5-3B-Instruct, 11.4G爆12G显存
# Qwen/Qwen2.5-3B-Instruct-AWQ, 10.4G显存
# Qwen/Qwen2.5-3B-Instruct-GGUF
# Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4， 10.2G显存
# Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8
# Qwen/Qwen2.5-7B-Instruct, 20G显存
# Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4, 11.3G爆12G显存
class Qwen:
    def __init__(self, model_name = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def infer(self, user_input, user_messages, chat_mode):
        # prompt 
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]['content'] = '你负责为一个语音聊天系统生成对话文本输出，使用长度接近的短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感。'
            else:
                with open('src/prompt.txt', 'r') as f:
                    user_messages[0]['content'] = f.read()
        user_messages.append({'role': 'user', 'content': user_input})
        print(user_messages)

        text = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        chat_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        user_messages.append({'role': 'assistant', 'content': chat_response})

        if len(user_messages) > 10:
            user_messages.pop(0)

        print(f'[Qwen] {chat_response}')
        return chat_response, user_messages


    def infer_stream(self, user_input, user_messages, llm_queue, chunk_size, chat_mode):
        print(f"[LLM] User input: {user_input}")
        time_cost = []
        start_time = time.time()
        # prompt 
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]['content'] = '你负责为一个语音聊天系统生成对话文本输出，使用短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感。'
            else:
                with open('src/prompt.txt', 'r', encoding='utf-8') as f:
                    user_messages[0]['content'] = f.read()
        print(f"[LLM] user_messages: {user_messages}")
        user_messages.append({'role': 'user', 'content': user_input})

        text = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, decode_kwargs={"errors": "ignore"})
        thread = Thread(target=self.model.generate, kwargs={**model_inputs, "streamer": streamer, "max_new_tokens": 512})
        thread.start()
        
        chat_response = ""
        buffer = ""
        sentence_buffer = ""
        sentence_split_pattern = re.compile(r'(?<=[,;.!?，；：。:！？》、”])')
        fp_flag = True
        print("[LLM] Start LLM streaming...")
        for chunk in streamer:
            #chat_response_chunk = chunk.choices[0].delta.content # 本地大模型返回的chunk就是str, 报错 An error occurred: 'str' object has no attribute 'choices'
            chat_response_chunk = chunk.replace("<|im_end|>", "").strip() # chunk结尾带"<|im_end|>", 去掉
            chat_response += chat_response_chunk
            buffer += chat_response_chunk

            sentences = sentence_split_pattern.split(buffer)
            
            if not sentences:
                continue
            
            for i in range(len(sentences) - 1):
                sentence = sentences[i].strip()
                sentence_buffer += sentence

                if fp_flag or len(sentence_buffer) >= chunk_size:
                    llm_queue.put(sentence_buffer)
                    time_cost.append(round(time.time()-start_time, 2))
                    start_time = time.time()
                    print(f"[LLM] Put into queue: {sentence_buffer}")
                    sentence_buffer = ""
                    fp_flag = False
            
            buffer = sentences[-1].strip()

        sentence_buffer += buffer
        if sentence_buffer:
            llm_queue.put(sentence_buffer)
            print(f"[LLM] Put into queue: {sentence_buffer}")

        llm_queue.put(None)
        
        user_messages.append({'role': 'assistant', 'content': chat_response})
        if len(user_messages) > 10:
            user_messages.pop(0)
        
        print(f"[LLM] Response: {chat_response}\n")
        
        return chat_response, user_messages, time_cost


class Qwen_API:
    def __init__(self, api_key = None, base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        api_key = api_key if api_key else os.getenv("DASHSCOPE_API_KEY")
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url,
        )
       
    def infer(self, user_input, user_messages, chat_mode):
        # prompt 
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]['content'] = '你负责为一个语音聊天系统生成对话文本输出，使用长度接近的短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感'
            else:
                with open('src/prompt.txt', 'r') as f:
                    user_messages[0]['content'] = f.read()
        user_messages.append({'role': 'user', 'content': user_input})
        print(user_messages)

        completion = self.client.chat.completions.create(
            model="qwen-turbo",
            messages=user_messages
        )
        print(completion)
        chat_response = completion.choices[0].message.content
        user_messages.append({'role': 'assistant', 'content': chat_response})

        if len(user_messages) > 10:
            user_messages.pop(0)
  
        print(f'[Qwen API] {chat_response}')
        return chat_response, user_messages


    def infer_stream(self, user_input, user_messages, llm_queue, chunk_size, chat_mode):
        print(f"[LLM] User input: {user_input}")
        time_cost = []
        start_time = time.time()
        # prompt 
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]['content'] = '你负责为一个语音聊天系统生成对话文本输出，使用短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感'
            else:
                with open('src/prompt.txt', 'r') as f:
                    user_messages[0]['content'] = f.read()
        print(f"[LLM] user_messages: {user_messages}")
        user_messages.append({'role': 'user', 'content': user_input})
        completion = self.client.chat.completions.create(
            model="qwen-turbo",
            messages=user_messages,
            stream=True
        )
        
        chat_response = ""
        buffer = ""
        sentence_buffer = ""
        sentence_split_pattern = re.compile(r'(?<=[,;.!?，；：。:！？》、”])')
        fp_flag = True
        print("[LLM] Start LLM streaming...")
        for chunk in completion:
            chat_response_chunk = chunk.choices[0].delta.content
            chat_response += chat_response_chunk
            buffer += chat_response_chunk

            sentences = sentence_split_pattern.split(buffer)
            
            if not sentences:
                continue
            
            for i in range(len(sentences) - 1):
                sentence = sentences[i].strip()
                sentence_buffer += sentence

                if fp_flag or len(sentence_buffer) >= chunk_size:
                    llm_queue.put(sentence_buffer)
                    time_cost.append(round(time.time()-start_time, 2))
                    start_time = time.time()
                    print(f"[LLM] Put into queue: {sentence_buffer}")
                    sentence_buffer = ""
                    fp_flag = False
            
            buffer = sentences[-1].strip()

        sentence_buffer += buffer
        if sentence_buffer:
            llm_queue.put(sentence_buffer)
            print(f"[LLM] Put into queue: {sentence_buffer}")

        llm_queue.put(None)
        
        user_messages.append({'role': 'assistant', 'content': chat_response})
        if len(user_messages) > 10:
            user_messages.pop(0)
        
        print(f"[LLM] Response: {chat_response}\n")
        
        return chat_response, user_messages, time_cost


if __name__ == "__main__":
    start_time = time.time()
    qwen = Qwen()
    print(f"Cost {time.time()-start_time} secs")
    start_time = time.time()
    qwen.infer_stream("讲一个长点的故事", [{'role': 'system', 'content': None}], None, None, "单轮对话 (一次性回答问题)")
    print(f"Cost {time.time()-start_time} secs")
