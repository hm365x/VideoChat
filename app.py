import subprocess
import sys
import os
import shutil
import gradio as gr
import modelscope_studio as mgr
import uvicorn
from fastapi import FastAPI
from modelscope import snapshot_download
import warnings
import time
import socket
import nltk

# 创空间部署需要
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings("ignore")

# os.environ["DASHSCOPE_API_KEY"] = "INPUT YOUR API KEY HERE"
os.environ["is_half"] = "True"

# 安装musetalk依赖
os.system('mim install mmengine')
os.system('mim install "mmcv==2.1.0"')
os.system('mim install "mmdet==3.2.0"')
# os.system('mim install "mmpose==1.2.0"') # for torch 2.1.2
os.system('mim install "mmpose==1.3.2"') # for torch 2.3.0
shutil.rmtree('./workspaces/results', ignore_errors= True)

# GLM-4-Voice 配置
sys.path.insert(0, "./src/GLM_4_Voice")
sys.path.insert(0, "./src/GLM_4_Voice/cosyvoice")
sys.path.insert(0, "./src/GLM_4_Voice//third_party/Matcha-TTS")

snapshot_download('ZhipuAI/glm-4-voice-tokenizer',cache_dir='./weights')
snapshot_download('ZhipuAI/glm-4-voice-decoder',cache_dir='./weights')
snapshot_download('ZhipuAI/glm-4-voice-9b',cache_dir='./weights')

from src.pipeline_llm import llm_pipeline
ENABLE_MLLM = True
if ENABLE_MLLM:
    from src.pipeline_mllm import mllm_pipeline


def is_port_open(host, port):
    """Check if a port is open on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout):
            return False


def wait_for_port(host, port, timeout=1800):
    """Wait for a port to open within a specified timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_open(host, port):
            print(f"Port {port} is open on {host}.")
            return True
        time.sleep(1)
    print(f"Timeout: Port {port} is not open on {host} after {timeout} seconds.")
    return False


def create_gradio():
    with gr.Blocks() as cascade_demo: 

        gr.Markdown(
            """
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            Chat with Digital Human (ASR-LLM-TTS-THG)
            </div>  
            <div style="text-align: center;">
               <a href="https://github.com/Henry-23/VideoChat"> GitHub </a> |
               <a href="https://mp.weixin.qq.com/s/jpoB8O2IyjhXeAWNWnAj7A"> 社区文章 </a>
            </div>

            """
        )
        with gr.Row():
            with gr.Column(scale = 2):
                user_chatbot = mgr.Chatbot(
                    label = "Chat History 💬",
                    value = [[None, {"text":"您好，请问有什么可以帮到您？您可以在下方的输入框点击麦克风录制音频或直接输入文本与我聊天。"}],],
                    avatar_images=[
                        {"avatar": os.path.abspath("data/icon/user.png")},
                        {"avatar": os.path.abspath("data/icon/qwen.png")},
                    ],
                    height= 500,
                    ) 

                with gr.Row():
                    avatar_name = gr.Dropdown(label = "数字人形象", choices = ["Avatar1 (通义万相)", "Avatar2 (通义万相)", "Avatar3 (MuseV)"], value = "Avatar1 (通义万相)")
                    chat_mode = gr.Dropdown(label = "对话模式", choices = ["单轮对话 (一次性回答问题)", "互动对话 (分多次回答问题)"], value = "单轮对话 (一次性回答问题)")
                    chunk_size = gr.Slider(label = "每次处理的句子最短长度", minimum = 0, maximum = 30, value = 5, step = 1) 
                    tts_module = gr.Dropdown(label = "TTS选型", choices = ["GPT-SoVits", "CosyVoice"], value = "CosyVoice")
                    avatar_voice = gr.Dropdown(label = "TTS音色", choices = ["longxiaochun (CosyVoice)", "longwan (CosyVoice)", "longcheng (CosyVoice)", "longhua (CosyVoice)", "少女 (GPT-SoVits)", "女性 (GPT-SoVits)", "青年 (GPT-SoVits)", "男性 (GPT-SoVits)"], value="longwan (CosyVoice)")
                    
                user_input = mgr.MultimodalInput(sources=["microphone"])

            with gr.Column(scale = 1):
                video_stream = gr.Video(label="Video Stream 🎬 (基于Gradio 5，可能卡顿，可参考左侧对话框生成的完整视频。)", streaming=True, height = 500, scale = 1)  
                user_input_audio = gr.Audio(label="音色克隆(可选项，输入音频控制在3-10s。如果不需要音色克隆，请清空。)", sources = ["microphone", "upload"],type = "filepath")
                stop_button = gr.Button(value="停止生成")

        # Use State to store user chat history
        user_messages = gr.State([{'role': 'system', 'content': None}])
        user_processing_flag = gr.State(False)
        lifecycle = mgr.Lifecycle()

        # voice clone
        user_input_audio.stop_recording(llm_pipeline.load_voice,
            inputs = [avatar_voice, tts_module, user_input_audio],
            outputs = [user_input])

        # loading TTS Voice
        avatar_voice.change(llm_pipeline.load_voice, 
            inputs=[avatar_voice, tts_module, user_input_audio], 
            outputs=[user_input]
            )
        lifecycle.mount(llm_pipeline.load_voice,
            inputs=[avatar_voice, tts_module, user_input_audio],
            outputs=[user_input]
        )

        # Submit
        user_input.submit(llm_pipeline.run_pipeline,
            inputs=[user_input, user_messages, chunk_size, avatar_name, tts_module, chat_mode, user_input_audio], 
            outputs=[user_messages]
            )
        user_input.submit(llm_pipeline.yield_results, 
            inputs=[user_input, user_chatbot, user_processing_flag],
            outputs = [user_input, user_chatbot, video_stream, user_processing_flag]
            )

        # refresh
        lifecycle.unmount(llm_pipeline.stop_pipeline, 
            inputs = user_processing_flag, 
            outputs = user_processing_flag
            )

        # stop
        stop_button.click(llm_pipeline.stop_pipeline, 
            inputs = user_processing_flag, 
            outputs = user_processing_flag
            )
    if not ENABLE_MLLM:
        return gr.TabbedInterface([cascade_demo], ['ASR-LLM-TTS-THG']).queue()
        
    with gr.Blocks() as mllm_demo:
          
        gr.Markdown(
            """
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            Chat with Digital Human (GLM-4-Voice - THG)
            </div>  
            <div style="text-align: center;">
               <a href="https://github.com/Henry-23/VideoChat"> GitHub </a> |
               <a href="https://mp.weixin.qq.com/s/jpoB8O2IyjhXeAWNWnAj7A"> 社区文章 </a>
            </div>
            """
        )
        with gr.Row():
            with gr.Column(scale = 2):
                user_chatbot = mgr.Chatbot(
                    label = "Chat History 💬",
                    value = [[None, {"text":"您好，请问有什么可以帮到您？您可以在下方的输入框点击麦克风录制音频或直接输入文本与我聊天。\n我使用了智谱AI开源的端到端语音模型GLM-4-Voice，您可以通过简单的指令控制情绪、生成方言等，例如：\n“用轻柔的声音引导我放松。”\n“用东北话介绍一下冬天有多冷。”\n“用北京话念一句绕口令。”\n"}],],
                    avatar_images=[
                        {"avatar": os.path.abspath("data/icon/user.png")},
                        {"avatar": os.path.abspath("data/icon/qwen.png")},
                    ],
                    height= 500,
                ) 

                with gr.Row():
                    avatar_name = gr.Dropdown(label = "数字人形象", choices = ["Avatar1 (通义万相)", "Avatar2 (通义万相)", "Avatar3 (MuseV)"], value = "Avatar1 (通义万相)")

                  
                user_input = mgr.MultimodalInput(sources=["microphone"])

            with gr.Column(scale = 1):
                video_stream = gr.Video(label="Video Stream 🎬 (基于Gradio 5，可能卡顿，可参考左侧对话框生成的完整视频。)", streaming=True, height = 500, scale = 1)  

        user_messages = gr.State("") #保存上一轮会话的token

        # GLM mode
        user_input.submit(
            mllm_pipeline.run_pipeline,
            inputs=[user_input, user_messages, avatar_name], 
            outputs=[user_messages]
            )
        user_input.submit(
            mllm_pipeline.yield_results, 
            inputs=[user_input, user_chatbot],
            outputs = [user_input, user_chatbot, video_stream]
            )
        

    return gr.TabbedInterface([cascade_demo, mllm_demo], ['ASR-LLM-TTS-THG', 'MLLM(GLM-4-Voice)-THG']).queue()

if __name__ == "__main__" and ENABLE_MLLM:
    # 启动 model_server
    model_server_process = subprocess.Popen(
        ['python', 'server.py'],
        cwd="./",
    )

    # 等待 model_server 启动并打开端口
    if wait_for_port('localhost', 10000):
        
        try:
            # warm up 
            mllm_pipeline.mllm.warm_up()
            # 启动 gradio demo
            print("Starting FastAPI with Gradio...")
            app = FastAPI()
            gradio_app = create_gradio()
            app = gr.mount_gradio_app(app, gradio_app, path='/')
            uvicorn.run(app, port=7860, log_level="warning")

            # 等待 model_server 进程结束
            model_server_process.wait()

        except KeyboardInterrupt:
            print("Terminating processes...")
            model_server_process.terminate()
    else:
        print("Failed to start model_server, terminating...")
        model_server_process.terminate()

elif __name__ == "__main__":
    # 启动 gradio demo
    print("Starting FastAPI with Gradio...")
    app = FastAPI()
    gradio_app = create_gradio()
    app = gr.mount_gradio_app(app, gradio_app, path='/')
    uvicorn.run(app, port=7860, log_level="warning")