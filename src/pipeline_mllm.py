import os
import sys
import argparse
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
import torch
import time
import numpy as np
import copy
import shutil
import threading
import queue
import time
import gradio as gr
import ffmpeg
import subprocess
import threading
from pydub import AudioSegment
import gradio as gr
import pandas as pd

from src.utils import get_timestamp_str, merge_videos, merge_frames_with_audio, get_video_duration
from src.glm import GLM_4_Voice
from src.thg import Muse_Talk
from src.pipeline_llm import llm_pipeline


# MLLM-THG
@torch.no_grad()
class MLLMPipeline:
    def __init__(self):
        print(f"Start initializing GLM-4-Voice")
        self.thg = llm_pipeline.thg
        self.mllm = GLM_4_Voice()
        print("[Done] Initialzation finished")

        self.timeout= 60 
        self.video_queue = queue.Queue()
        self.mllm_queue = queue.Queue()
        self.thg_queue = queue.Queue()

        self.chat_history = []
        self.stop = threading.Event()


    # GLM-4-Voice暂时没提供明确的停止方式
    def stop_pipeline(self):
        pass
        

    def flush_pipeline(self):
        print("Flushing pipeline....")
        self.video_queue = queue.Queue()
        self.mllm_queue = queue.Queue()
        self.thg_queue = queue.Queue()
        self.chat_history = []
        self.thg.idx = 0
        self.start_time = None
        

    def run_pipeline(self, user_input, user_messages, avatar_name):
        self.flush_pipeline()
        self.start_time = time.time()
        avatar_name = avatar_name.split(" ")[0]
        project_path = f"./workspaces/results/{avatar_name}/{get_timestamp_str()}"
        os.makedirs(project_path, exist_ok=True)

        # Start pipeline
        gr.Info("Start processing.", duration = 2)
        try:
            # warm up
            self.thg_thread = threading.Thread(target=self.thg_worker, args=(project_path, avatar_name, ))
            self.thg_thread.start()

            self.ffmpeg_thread = threading.Thread(target=self.ffmpeg_worker)
            self.ffmpeg_thread.start()

            # MLLM streaming out
            user_messages = self.mllm.infer(project_path, user_input, user_messages, self.mllm_queue)

            self.thg_thread.join()
            self.ffmpeg_thread.join()

            # Stop pipeline
            # if self.stop.is_set():
            #     print("Stop pipeline......")
            # else:
            #     print("Finish pipeline......")

            return user_messages

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            gr.Error(f"An error occurred: {str(e)}")
            return ""

 
    def get_time_cost(self):
        index = [str(i) for i in range(len(self.time_cost[0]))]
        total_time = [round(sum(x), 2) for x in zip(*self.time_cost[1:])]
        self.time_cost.append(total_time)

        s = "Index     Duration     MLLM     THG       ffmpeg    Cost\n"

        for row in zip(index, *self.time_cost):
            s += "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(*row)

        return s


    def yield_results(self, user_input, user_chatbot):
        user_chatbot.append([
            {
                "text": user_input.text,
                "files": user_input.files,
            },
            {
                "text": "开始生成......\n",
            }
        ])
        yield gr.update(interactive=False, value=None), user_chatbot, None

        time.sleep(1)
        index = 0
        videos_path = None
        fp_latency = 0
        start_time = time.time()
        print("[Listener] Start yielding results from queue.")

        try:
            while not self.stop.is_set():
                try:
                    video_path = self.video_queue.get(timeout=1)
                    if not video_path:
                        break
                    if index == 0:
                        fp_latency = time.time() - self.start_time
                    videos_path = os.path.dirname(video_path)
                    user_chatbot[-1][1]["text"]+=self.chat_history[index]

                    yield gr.update(interactive=False, value=None), user_chatbot, video_path
                    gr.Info(f"Streaming video_{index} from queue.", duration = 1)
                    print(f"[Listener] Streaming video_{index} from queue.")
                    time.sleep(2)
                    index += 1
                    start_time = time.time()
                    
                except queue.Empty: 
                    if time.time() - start_time > self.timeout:
                        gr.Info("Timeout, stop listening video stream queue.")
                        break

                except Exception as e:
                    gr.Error(f"An error occurred: {str(e)}")


            # Merge all videos
            if not self.stop.is_set() and videos_path:
                merged_video_path = merge_videos(videos_path)
                # video mp4 format
                llm_response_txt = user_chatbot[-1][1]["text"]  + f"""<video src="{merged_video_path}"></video>\n""" 
                # First Packet RT
                llm_response_txt = llm_response_txt + f"首包延迟：{round(fp_latency, 2)}s\n"
                user_chatbot[-1][1] = {
                        "text": llm_response_txt,
                        "flushing": False
                    }

            if self.stop.is_set():
                user_chatbot[-1][1]["text"]+="\n停止生成，请稍等......"
            

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            gr.Error(f"An error occurred: {str(e)}")

        finally:
            yield gr.update(interactive=True, value=None), user_chatbot, None

            if videos_path: 
                results_path = os.path.dirname(videos_path)
                print(f"Remove results: {results_path}")
                shutil.rmtree(results_path, ignore_errors=True)


    def thg_worker(self, project_path, avatar_name):
        # 在本线程中提前做一次推理，避免第一次推理产生的额外初始化耗时
        gr.Info("Warming up THG Module...", duration = 2)
        self.thg.warm_up()

        start_time = time.time()
        index = 0
        while not self.stop.is_set():
            try:
                llm_response_text, llm_response_audio = self.mllm_queue.get(timeout=1)
                if not llm_response_text and not llm_response_audio:
                    break

                print(f"[THG] Get audio from mllm_queue: {llm_response_text}, {llm_response_audio}")
                self.chat_history.append(llm_response_text)
                infer_start_time = time.time()
                self.thg.infer(project_path=project_path, audio_path=llm_response_audio, avatar_name=avatar_name)
                # self.time_cost[2].append(round(time.time()-infer_start_time,2))
                self.thg_queue.put(llm_response_audio)
                start_time = time.time()
                index+=1

            except queue.Empty:
                if time.time() - start_time > self.timeout:
                    gr.Info("THG Timeout")
                    break
                
        self.thg_queue.put(None)


    def ffmpeg_worker(self):
        start_time = time.time()
        index = 0
        while not self.stop.is_set():
            try:
                llm_response_audio = self.thg_queue.get(timeout=1)
                print(f"[FFMPEG] Get frames from thg_queue: {llm_response_audio}")
                if not llm_response_audio:
                    break
                infer_start_time = time.time()
                video_result = merge_frames_with_audio(llm_response_audio)
                # self.time_cost[3].append(round(time.time()-infer_start_time,2))
                self.video_queue.put(video_result)
                # self.time_cost[0].append(get_video_duration(video_result))

                start_time = time.time()
                index+=1
            except queue.Empty:
                if time.time() - start_time > self.timeout:
                    gr.Info("ffmpeg Timeout")
                    break
                
        self.video_queue.put(None)

# 实例化         
mllm_pipeline = MLLMPipeline()
