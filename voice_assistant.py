import json
import time
import queue
import threading
import re
from pathlib import Path
import requests
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class VoiceAssistant:
    def __init__(self, token: str, system_prompt: str = None):
        self.token = token
        self.conversation_history = []
        
        # 初始化系统提示语
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
        
        self.max_history = 5
        self.speech_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # 添加 TTS 任务队列
        self.tts_tasks = queue.Queue()
        self.speech_files = {}  # 存储音频文件路径
        self.current_tts_index = 0  # 当前TTS任务序号
        self.current_play_index = 0  # 当前播放序号
        
        self.client = OpenAI(
            api_key=token,
            base_url="https://api.siliconflow.cn/v1"
        )
        
        # 音频参数
        self.sample_rate = 16000
        self.silence_threshold = 0.02
        self.silence_duration = 1.5
        self.chunk_size = 1024
        self.recording_data = []

    def audio_callback(self, indata, frames, time, status):
        """音频回调函数"""
        self.recording_data.append(indata.copy())

    def is_silent(self, audio_data):
        """检查音频块是否为静音"""
        return np.max(np.abs(audio_data)) < self.silence_threshold

    def record_audio(self) -> str:
        """智能录音函数"""
        print("开始录音...（持续静音1.5秒后自动停止）")
        self.recording_data = []
        
        with sd.InputStream(callback=self.audio_callback,
                          channels=1,
                          samplerate=self.sample_rate,
                          blocksize=self.chunk_size):
            
            silence_start = None
            while True:
                if len(self.recording_data) == 0:
                    time.sleep(0.1)
                    continue

                latest_data = self.recording_data[-1]
                
                if self.is_silent(latest_data):
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= self.silence_duration:
                        break
                else:
                    silence_start = None

        print("录音完成")
        
        audio_data = np.concatenate(self.recording_data, axis=0)
        audio_file = "input_audio.wav"
        sf.write(audio_file, audio_data, self.sample_rate)
        
        return audio_file

    def transcribe_audio(self, audio_file: str) -> str:
        """使用 API 进行语音识别"""
        print("正在转写语音...")
        
        url = "https://api.siliconflow.cn/v1/audio/transcriptions"
        
        files = {
            'file': ('audio.wav', open(audio_file, 'rb'), 'audio/wav'),
            'model': (None, 'FunAudioLLM/SenseVoiceSmall')
        }
        
        headers = {
            "Authorization": f"Bearer {self.token}"
        }

        response = requests.post(url, headers=headers, files=files)
        
        if response.status_code == 200:
            result = response.json()
            transcribed_text = result.get('text', '')
            print(f"语音转写结果: {transcribed_text}")
            return transcribed_text
        else:
            print(f"语音识别失败: {response.text}")
            return ""

    def text_to_speech_worker(self, text: str, index: int):
        """TTS 工作线程"""
        speech_file_path = Path(f"output_audio_{time.time()}.mp3")
        
        try:
            with self.client.audio.speech.with_streaming_response.create(
                model="FunAudioLLM/CosyVoice2-0.5B",
                voice="FunAudioLLM/CosyVoice2-0.5B:alex",
                input=text,
                response_format="mp3"
            ) as response:
                response.stream_to_file(speech_file_path)
            
            # 将文件路径存储到字典中
            self.speech_files[index] = speech_file_path
            # 通知播放线程
            self.speech_queue.put(index)
        except Exception as e:
            print(f"TTS 转换错误: {e}")

    def play_audio_worker(self):
        """音频播放工作线程"""
        while True:
            try:
                # 获取要播放的音频索引
                index = self.speech_queue.get()
                
                # 如果不是当前应该播放的索引，重新放回队列
                if index != self.current_play_index:
                    self.speech_queue.put(index)
                    time.sleep(0.1)
                    continue
                
                # 获取对应的音频文件路径
                speech_file_path = self.speech_files.get(index)
                if speech_file_path and speech_file_path.exists():
                    data, samplerate = sf.read(speech_file_path)
                    sd.play(data, samplerate)
                    sd.wait()
                    # 删除已播放的文件
                    speech_file_path.unlink(missing_ok=True)
                    # 从字典中移除
                    self.speech_files.pop(index, None)
                    # 更新播放索引
                    self.current_play_index += 1
                
            except Exception as e:
                print(f"音频播放错误: {e}")
            finally:
                self.speech_queue.task_done()

    def chat_with_llm(self, text: str) -> str:
        """与 LLM 对话"""
        self.conversation_history.append({
            "role": "user",
            "content": text
        })
        
        if len(self.conversation_history) > self.max_history * 2 + 1:
            system_prompt = self.conversation_history[0] if self.conversation_history[0]["role"] == "system" else None
            recent_messages = self.conversation_history[-(self.max_history * 2):]
            self.conversation_history = [system_prompt] + recent_messages if system_prompt else recent_messages

        payload = {
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "messages": self.conversation_history,
            "stream": True,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7
        }
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://api.siliconflow.cn/v1/chat/completions",
            json=payload,
            headers=headers,
            stream=True
        )

        collected_messages = []
        buffer = ""
        self.current_tts_index = 0  # 重置TTS任务序号
        self.current_play_index = 0  # 重置播放序号

        for line in response.iter_lines():
            if line:
                if line.startswith(b"data: "):
                    json_str = line.decode('utf-8')[6:]
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        json_data = json.loads(json_str)
                        content = json_data['choices'][0]['delta'].get('content', '')
                        if content:
                            buffer += content
                            sentences = self.split_into_sentences(buffer)
                            if sentences:
                                complete_sentences = sentences[:-1]
                                buffer = sentences[-1]
                                
                                for sentence in complete_sentences:
                                    if sentence.strip():
                                        collected_messages.append(sentence)
                                        # 提交TTS任务时包含序号
                                        self.executor.submit(
                                            self.text_to_speech_worker, 
                                            sentence,
                                            self.current_tts_index
                                        )
                                        self.current_tts_index += 1
                                        print(sentence, end='', flush=True)
                    except json.JSONDecodeError:
                        continue

        if buffer.strip():
            collected_messages.append(buffer)
            self.executor.submit(
                self.text_to_speech_worker,
                buffer,
                self.current_tts_index
            )
            print(buffer, end='', flush=True)

        complete_response = "".join(collected_messages)
        self.conversation_history.append({
            "role": "assistant",
            "content": complete_response
        })
        
        return complete_response

    def split_into_sentences(self, text: str) -> list:
        """将文本分割成句子"""
        sentences = re.split('([，。！？.!?])', text)
        result = []
        for i in range(0, len(sentences)-1, 2):
            if sentences[i]:
                result.append(sentences[i] + (sentences[i+1] if i+1 < len(sentences) else ""))
        if len(sentences) % 2 == 1 and sentences[-1]:
            result.append(sentences[-1])
        return result

    def run(self):
        """运行语音助手"""
        play_thread = threading.Thread(target=self.play_audio_worker, daemon=True)
        play_thread.start()

        print("语音助手已启动（按 Ctrl+C 退出）")
        try:
            while True:
                input("按回车开始录音 (说话时停顿1秒自动结束)...")
                audio_file = self.record_audio()
                text = self.transcribe_audio(audio_file)
                
                if text.strip():
                    print(f"\n用户: {text}")
                    print("\n助手: ", end='', flush=True)
                    response = self.chat_with_llm(text)
                    print("\n")
                else:
                    print("未检测到语音输入，请重试。")
                
        except KeyboardInterrupt:
            print("\n程序已退出")
        finally:
            self.executor.shutdown(wait=False)

def main():
    # 配置部分
    token = ""  # 替换为你的 API token
    
    # 示例系统提示语
    system_prompt = """你是一个 AI 助手。
    你的回答应该：
    1. 简洁明了
    2. 富有同理心
    3. 准确专业
    4. 在合适的时候适当幽默
    请用自然、流畅的语气与用户交谈。
    你需要多输出句号、问号、感叹号等标点符号，以使回答更加自然。"""
    
    # 初始化助手
    assistant = VoiceAssistant(token=token, system_prompt=system_prompt)
    assistant.run()

if __name__ == "__main__":
    main()