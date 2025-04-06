#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import torch
import soundfile as sf
import logging
from datetime import datetime
import platform
import sys
import warnings
import glob
import re
from pathlib import Path

# 用于ASR的依赖，如果没有安装，将提示用户安装
try:
    import whisper
    has_whisper = True
except ImportError:
    has_whisper = False
    warnings.warn("未检测到 OpenAI Whisper 库，将无法自动转换音频为文本。使用 'pip install -U openai-whisper' 安装。")

# 确保可以导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = current_dir  # 假设脚本在项目根目录
sys.path.insert(0, root_dir)

from cli.SparkTTS import SparkTTS

def transcribe_audio(audio_path):
    """使用Whisper模型将音频转换为文本"""
    if not has_whisper:
        logging.warning("未安装Whisper库，无法自动转换音频为文本")
        return None
    
    try:
        logging.info(f"正在使用Whisper识别音频文件: {audio_path}")
        # 加载小型模型以提高速度，你也可以使用"base"或"large"获得更高准确率
        model = whisper.load_model("small")
        
        # 转录音频
        result = model.transcribe(audio_path, language="zh")
        transcribed_text = result["text"]
        
        logging.info(f"音频识别结果: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        logging.error(f"音频识别过程中出现错误: {str(e)}")
        return None

def check_and_fix_model(model_dir):
    """
    检查模型文件并尝试修复常见问题
    返回: (bool, str) - 是否需要修复，错误信息
    """
    try:
        # 检查模型目录是否存在
        if not os.path.exists(model_dir):
            return False, f"模型目录不存在: {model_dir}"
        
        # 检查是否有.bin或.pt文件
        model_files = glob.glob(os.path.join(model_dir, "*.bin")) + \
                     glob.glob(os.path.join(model_dir, "*.pt"))
        
        if not model_files:
            return False, f"未找到模型文件 (.bin 或 .pt): {model_dir}"
        
        logging.info(f"找到以下模型文件: {', '.join([os.path.basename(f) for f in model_files])}")
        
        # 尝试加载模型文件检查是否完整
        try:
            # 只检查，不实际加载到内存
            checkpoint = torch.load(model_files[0], map_location="cpu")
            logging.info(f"模型加载检查通过: {os.path.basename(model_files[0])}")
            return True, "模型文件检查通过"
        except Exception as e:
            logging.error(f"模型文件损坏或不完整: {str(e)}")
            return False, f"模型文件损坏或不完整: {str(e)}"
            
    except Exception as e:
        return False, f"模型检查出错: {str(e)}"

def main():
    """更易于调试的Spark-TTS推理入口脚本"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # 默认参数
    default_text = "俩人去森林里打猎，每人枪里面只有5颗子弹，突然天上飞来一个老鹰，表弟上来就是一枪，当然太远没打到，接着往前走，兄弟俩又看到了一个野鸡，表弟立马举枪，上来就是一枪，可惜又空了。接着兄弟俩往森林深处走去， 突然从面前窜过去一头蓬蓬， 哥俩立马跟了上去，表弟眼疾手快，举枪朝着蓬蓬屁股便是一枪，可惜打歪了，又开了一枪，谁知这个蓬蓬带甲，就是所谓的山猪带甲，老虎不怕，第二枪打中了，但是甲太厚实，没造成致命伤害，还是让蓬蓬跑掉了。表弟手上还剩下一颗子弹，大表哥贼狗，一枪未发。表哥说没有百分之百的把握不要乱开枪，惊了其他动物， 我们将损失今天的打猎机会。 继续往深林走去，突然看到池塘边一个草丛中有个小白兔，温柔又可爱，表弟说，这次我有把握，马步一扎，屏气凝神，瞄着小白兔开了一枪。只见前方草丛传来骂骂咧咧的声响： 谁他妈的不讲武德，竟然搞偷袭。表弟跑过去又补了两拳，边打边骂：让你多嘴，然后结果了小白兔光辉灿烂的一生。 表弟自然是十分高兴，今天没有空军。表哥也表示羡慕，不停滴夸表弟枪法炉火纯青、不念私情、手起刀落，难得大才。 然后两兄弟就互相吹捧着继续往森林深处走去，只见森林阴森诡异、魅影幢幢，突然两人眼前一黑，一个黑黢黢的鬼魅从他俩面前窜了过去，躲在前方荆棘之后，表弟大气不敢滴对着表哥：表哥，一闪而过的黑影是啥？ 表哥说：是黑山魈，最爱吃提着兔子的老百姓了！ 表弟立马把别在腰间的小白兔扔到地上了。 表哥嘿嘿一笑：骗你呢，你也信。 就在这时，从树后窜出一个长着血盆大口 面目狰狞的黑旋风直撞撞地朝着表哥扑来，说时迟那时快，表哥吓得一个趔趄，躲得了这个黑旋风的冲撞，这才捡了一条小命，话说这个黑旋风就是大名鼎鼎的黑瞎子。  黑瞎子偷袭铺了一个空，气急败坏调转身体，凶狠无比看着兄弟俩，满眼全是人肉烤串，牙齿磨的滋滋冒火星子。 表哥慌乱地子弹上膛，叩开保险，举枪瞄着黑瞎子，虽然有真理在手，计算了一下盈亏比，面对黑瞎子最多也就五五开，大概率会同归于尽。黑瞎子见多识广也不傻，知道面前这个小鬼手里拿家伙的威力， 心想要赢只能以速度、力量和技巧取胜， 黑瞎子先是往左跳了一下，躲开面前的两个黑点冒出的火星子，然后再往右跳跃了一步，又躲开了黑点爆发的火星子， 黑瞎子想来个声东击西、欲擒故纵、高抛低吸、追涨杀跌、杠杆期权，各种套路演示一遍， 最后朝着表哥扑了过来，表哥对着表弟说：不要高位站岗，快跑，崩盘了。 然后表哥闭上眼，朝着前方连开三枪，清空了所有的子弹。 再次睁开眼睛时，发现黑瞎子七窍流血， 气绝身亡、眼睛怒视表哥，死不瞑目。两兄弟后知后觉，一屁股瘫坐地上， 表弟说：表哥，你真牛逼，如此危急之下，枪法和索罗斯一样精准。 表哥说：还好福大命大，运气好， 打中了它。第二天，兄弟俩又来森林打猎了，欲知后事如何，请听下回讲解………   以上出自《李子-劝学》"
    default_prompt_text = "俩人去森林里打猎，每人枪里面只有5颗子弹，突然天上飞来一个老鹰，表弟上来就是一枪，当然太远没打到，接着往前走，兄弟俩又看到了一个野鸡，表弟立马举枪，上来就是一枪，可惜又空了。"
    default_prompt_speech_path = "example/neo.wav"
    default_model_dir = "pretrained_models/Spark-TTS-0.5B"
    default_save_dir = "example/results"
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Spark-TTS语音合成调试脚本")
    
    parser.add_argument("--model_dir", type=str, default=default_model_dir,
                        help="模型目录路径")
    parser.add_argument("--save_dir", type=str, default=default_save_dir,
                        help="生成音频保存目录")
    parser.add_argument("--device", type=int, default=0,
                        help="设备号(CUDA/MPS)")
    parser.add_argument("--force_cpu", action="store_true",
                        help="强制使用CPU，绕过GPU加速")
    parser.add_argument("--text", type=str, default=default_text,
                        help="要合成的文本")
    parser.add_argument("--prompt_text", type=str, default=default_prompt_text,
                        help="提示音频的文本，如不提供则尝试自动识别")
    parser.add_argument("--prompt_speech_path", type=str, default=default_prompt_speech_path,
                        help="提示音频文件路径")
    parser.add_argument("--gender", type=str, choices=["male", "female"],
                        help="性别选择")
    parser.add_argument("--pitch", type=str, 
                        choices=["very_low", "low", "moderate", "high", "very_high"],
                        help="音高选择")
    parser.add_argument("--speed", type=str,
                        choices=["very_low", "low", "moderate", "high", "very_high"],
                        help="语速选择")
    
    args = parser.parse_args()
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    # 设置设备
    if args.force_cpu:
        device = torch.device("cpu")
        logging.info("强制使用CPU以避免兼容性问题")
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        # 在Apple Silicon上尝试使用CPU先检查模型，如果有问题再尝试MPS
        device = torch.device("cpu")
        logging.info("在Apple Silicon上使用CPU以避免MPS兼容性问题")
    elif torch.cuda.is_available():
        # System with CUDA support
        device = torch.device(f"cuda:{args.device}")
        logging.info(f"使用CUDA设备: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU加速不可用，使用CPU")
    
    # 以下是原始设备选择逻辑，现已注释掉
    # if platform.system() == "Darwin" and torch.backends.mps.is_available():
    #     # macOS with MPS support (Apple Silicon)
    #     device = torch.device(f"mps:{args.device}")
    #     logging.info(f"使用MPS设备: {device}")
    # elif torch.cuda.is_available():
    #     # System with CUDA support
    #     device = torch.device(f"cuda:{args.device}")
    #     logging.info(f"使用CUDA设备: {device}")
    # else:
    #     # Fall back to CPU
    #     device = torch.device("cpu")
    #     logging.info("GPU加速不可用，使用CPU")
    
    try:
        # 先检查模型文件
        logging.info(f"检查模型文件: {args.model_dir}")
        status, message = check_and_fix_model(args.model_dir)
        if not status:
            logging.error(f"模型检查失败: {message}")
            logging.error("请确保模型文件已正确下载并完整。")
            logging.info("尝试继续加载模型，但可能会失败...")
        
        # 初始化模型
        logging.info(f"正在从 {args.model_dir} 加载模型...")
        try:
            model = SparkTTS(args.model_dir, device)
        except RuntimeError as e:
            if "Missing tensor: mel_transformer.mel_scale.fb" in str(e):
                logging.error("遇到已知的'Missing tensor: mel_transformer.mel_scale.fb'错误")
                logging.error("这通常是由于模型版本与代码不兼容或者在Apple Silicon上的兼容性问题")
                logging.error("尝试以下解决方案:")
                logging.error("1. 使用 --force_cpu 参数强制使用CPU")
                logging.error("2. 确保使用了正确版本的模型文件")
                logging.error("3. 在非Apple Silicon设备上运行")
                return 1
            else:
                raise e
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(args.save_dir, f"{timestamp}.wav")
        
        logging.info("开始推理...")
        logging.info(f"输入文本: {args.text}")
        
        # 检查是否需要自动识别音频文本
        prompt_text = args.prompt_text
        if prompt_text is None and os.path.exists(args.prompt_speech_path):
            logging.info("未提供提示音频文本，尝试自动识别...")
            prompt_text = transcribe_audio(args.prompt_speech_path)
            if prompt_text is None:
                prompt_text = default_prompt_text
                logging.warning(f"自动识别失败，使用默认文本: {prompt_text}")
            else:
                logging.info(f"成功识别音频文本: {prompt_text}")
        elif prompt_text is None:
            prompt_text = default_prompt_text
            logging.warning(f"提示音频文件不存在或未提供文本，使用默认文本: {prompt_text}")
        
        # 执行推理
        with torch.no_grad():
            # 为调试需要，可在此处设置断点
            wav = model.inference(
                args.text,
                args.prompt_speech_path,
                prompt_text=prompt_text,
                gender=args.gender,
                pitch=args.pitch,
                speed=args.speed,
            )
            
            # 保存音频
            sf.write(save_path, wav, samplerate=16000)
        
        logging.info(f"音频已保存至: {save_path}")
        
    except Exception as e:
        logging.error(f"推理过程中发生错误: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())