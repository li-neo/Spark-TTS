#!/usr/bin/env python3
# Copyright (c) 2025 SparkAudio
# 修复后的启动脚本

import sys
import argparse
from fix_gradio import fix_gradio_client_utils

def parse_arguments():
    """
    解析命令行参数，如模型目录和设备ID。
    """
    parser = argparse.ArgumentParser(description="Spark TTS Gradio server.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the GPU device to use (e.g., 0 for cuda:0)."
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server host/IP for Gradio app."
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7861,
        help="Server port for Gradio app."
    )
    return parser.parse_args()

if __name__ == "__main__":
    print("正在修复 gradio_client 库中的错误...")
    if fix_gradio_client_utils():
        print("修复成功，正在启动 Web UI...")
        # 导入 webui 模块
        from webui import build_ui
        
        # 解析命令行参数
        args = parse_arguments()
        
        # 构建 Gradio 演示，指定模型目录和 GPU 设备
        demo = build_ui(
            model_dir=args.model_dir,
            device=args.device
        )
        
        # 启动 Gradio，指定服务器名称和端口，并启用分享功能
        demo.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=True
        )
    else:
        print("无法修复 gradio_client 错误，请联系开发人员获取支持。")
        sys.exit(1)