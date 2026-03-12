#!/usr/bin/env python3
"""
LaTeX OCR 全流程可视化 Web 服务器
================================
启动方式: python visualizer.py
然后在浏览器中打开 http://localhost:8080
"""

import base64
import io
import json
import os
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import numpy as np
from PIL import Image

# 导入 ONNX 推理模块
from latex_ocr_visualizer import (
    Config, ONNXModels, Tokenizer, ImagePreprocessor,
    LaTeXInferenceEngine, format_latex
)

# 尝试导入 onnxruntime
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("错误: onnxruntime 未安装")

# ============ 配置 ============
PORT = 8080
MODEL_DIR = Path(__file__).parent

# ============ HTML 模板 ============
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LaTeX OCR 全流程可视化</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        
        .header {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 24px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .container {
            display: flex;
            flex-wrap: wrap;
            padding: 20px;
            gap: 20px;
            justify-content: center;
            align-items: flex-start;
        }
        
        .drop-zone {
            width: 400px;
            height: 300px;
            border: 3px dashed #3a7bd5;
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255,255,255,0.05);
            position: relative;
            overflow: hidden;
        }
        
        .drop-zone:hover, .drop-zone.dragover {
            border-color: #00d2ff;
            background: rgba(0,210,255,0.1);
            transform: scale(1.02);
        }
        
        .drop-zone.has-image {
            border-style: solid;
            border-color: #4CAF50;
        }
        
        .drop-zone-text {
            font-size: 18px;
            color: #aaa;
            pointer-events: none;
        }
        
        .drop-zone-hint {
            font-size: 14px;
            color: #666;
            margin-top: 10px;
            pointer-events: none;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: none;
        }
        
        .pipeline {
            flex: 1;
            min-width: 300px;
            max-width: 600px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
        }
        
        .pipeline-title {
            font-size: 18px;
            margin-bottom: 15px;
            color: #00d2ff;
        }
        
        .step {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 15px;
            opacity: 0.5;
            transition: all 0.3s ease;
        }
        
        .step.active {
            opacity: 1;
            background: rgba(0,210,255,0.2);
            border-left: 4px solid #00d2ff;
        }
        
        .step.completed {
            opacity: 1;
            background: rgba(76,175,80,0.2);
            border-left: 4px solid #4CAF50;
        }
        
        .step-icon {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #333;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        
        .step.active .step-icon {
            background: #00d2ff;
            animation: pulse 1s infinite;
        }
        
        .step.completed .step-icon {
            background: #4CAF50;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .step-content {
            flex: 1;
        }
        
        .step-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .step-desc {
            font-size: 12px;
            color: #aaa;
        }
        
        .result-zone {
            width: 100%;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            order: 10;
        }
        
        .result-title {
            font-size: 16px;
            margin-bottom: 10px;
            color: #4CAF50;
        }
        
        .result-content {
            font-family: 'Consolas', monospace;
            font-size: 18px;
            background: #1a1a2e;
            padding: 15px;
            border-radius: 10px;
            word-break: break-all;
            min-height: 30px;
        }
        
        .result-content .cursor {
            display: inline-block;
            width: 2px;
            height: 20px;
            background: #00d2ff;
            animation: blink 1s infinite;
            vertical-align: middle;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        
        .image-preview {
            width: 100%;
            max-height: 200px;
            object-fit: contain;
            border-radius: 10px;
            margin-top: 10px;
            display: none;
        }
        
        .step.has-preview .image-preview {
            display: block;
        }
        
        .preprocess-card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            transition: all 0.3s ease;
            opacity: 0;
            transform: translateY(20px);
        }
        
        .preprocess-card.show {
            opacity: 1;
            transform: translateY(0);
        }
        
        .preprocess-card img {
            width: 120px;
            height: 120px;
            object-fit: contain;
            border-radius: 8px;
            background: #000;
        }
        
        .preprocess-card .step-name {
            font-size: 12px;
            margin-top: 8px;
            color: #aaa;
        }
        
        .preprocess-card .arrow {
            font-size: 24px;
            color: #00d2ff;
            align-self: center;
        }
        
        .encoder-card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            opacity: 0;
            transform: scale(0.8);
            transition: all 0.5s ease;
        }
        
        .encoder-card.show {
            opacity: 1;
            transform: scale(1);
        }
        
        .encoder-card img {
            width: 200px;
            height: 200px;
            border-radius: 8px;
            background: #000;
        }
        
        .encoder-card .step-name {
            font-size: 14px;
            margin-top: 10px;
            color: #aaa;
        }
        
        .feature-cell {
            width: 28px;
            height: 28px;
            background: #222;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 6px;
            color: #666;
            transition: all 0.3s ease;
            border-radius: 2px;
            cursor: pointer;
        }
        
        .feature-cell.active {
            background: #444;
            color: #fff;
            transform: scale(1.1);
            box-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
        }
        
        .feature-cell.colored {
            color: #fff;
            text-shadow: 0 0 2px #000;
        }
        
        .feature-cell.fully-active {
            animation: cellGlow 0.5s ease-out;
        }
        
        @keyframes cellGlow {
            0% { transform: scale(1.3); box-shadow: 0 0 20px rgba(255, 100, 0, 0.8); }
            100% { transform: scale(1); box-shadow: none; }
        }
        
        .heatmap-zoom {
            animation: heatmapScan 2s ease-out forwards;
        }
        
        @keyframes heatmapScan {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        
        .heatmap-cell {
            width: 16px;
            height: 16px;
            background: #111;
            opacity: 0;
            transition: all 0.1s ease;
        }
        
        .heatmap-cell.show {
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 LaTeX OCR 全流程可视化</h1>
    </div>
    
    <div class="container">
        <div class="drop-zone" id="dropZone">
            <span class="drop-zone-text">📁 拖拽图片到这里</span>
            <span class="drop-zone-hint">或点击选择图片</span>
            <img class="preview-image" id="previewImage" alt="预览">
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
        </div>
        
        <div class="pipeline">
            <div class="pipeline-title">📊 处理流程</div>
            
            <div class="step" id="step1">
                <div class="step-icon">🖼️</div>
                <div class="step-content">
                    <div class="step-title">1. 加载图像</div>
                    <div class="step-desc">读取用户拖入的图片文件</div>
                </div>
            </div>
            
            <div class="step" id="step2">
                <div class="step-icon">🔄</div>
                <div class="step-content">
                    <div class="step-title">2. 图像预处理</div>
                    <div class="step-desc">灰度转换、反转、裁剪、缩放</div>
                </div>
            </div>
            
            <div class="step" id="step3">
                <div class="step-icon">🔢</div>
                <div class="step-content">
                    <div class="step-title">3. 特征提取 (Encoder)</div>
                    <div class="step-desc">HGNetv2 编码为 144×2048 特征图</div>
                </div>
            </div>
            
            <div class="step" id="step4">
                <div class="step-icon">🤖</div>
                <div class="step-content">
                    <div class="step-title">4. 自回归生成 (Decoder)</div>
                    <div class="step-desc">MBART 解码为 LaTeX token 序列</div>
                </div>
            </div>
            
            <div class="step" id="step5">
                <div class="step-icon">📝</div>
                <div class="step-content">
                    <div class="step-title">5. 后处理</div>
                    <div class="step-desc">Token 映射、格式清理</div>
                </div>
            </div>
        </div>
        
        <div class="result-zone" id="resultZone" style="display: none;">
            <div class="result-title">识别结果</div>
            <div class="result-content" id="resultContent"></div>
        </div>
        
        <!-- 预处理可视化 -->
        <div class="preprocess-viz" id="preprocessViz" style="display: none; width: 100%; margin-top: 20px;">
            <div class="pipeline-title">图像预处理步骤</div>
            <div class="preprocess-steps" id="preprocessSteps" style="display: flex; gap: 15px; flex-wrap: wrap; justify-content: center;"></div>
        </div>
        
        <!-- 归一化过程可视化 -->
        <div class="norm-viz" id="normViz" style="display: none; width: 100%; margin-top: 20px;">
            <div class="pipeline-title">归一化过程</div>
            
            <!-- 归一化前 -->
            <div class="norm-step" id="normStep1" style="margin-top: 15px; text-align: center;">
                <div class="step-desc" style="color: #aaa; margin-bottom: 10px;">归一化前 (0-255)</div>
                <div id="normBefore" style="font-family: monospace; font-size: 14px; background: #222; padding: 15px; border-radius: 10px; display: inline-block;"></div>
            </div>
            
            <!-- 归一化计算 -->
            <div class="norm-step" id="normStep2" style="margin-top: 20px; text-align: center;">
                <div class="step-desc" style="color: #aaa; margin-bottom: 10px;">归一化计算</div>
                <div id="normFormula" style="font-family: monospace; font-size: 16px; background: #222; padding: 15px; border-radius: 10px; display: inline-block; color: #00d2ff;"></div>
            </div>
            
            <!-- 归一化后 -->
            <div class="norm-step" id="normStep3" style="margin-top: 20px; text-align: center;">
                <div class="step-desc" style="color: #aaa; margin-bottom: 10px;">归一化后</div>
                <div id="normAfter" style="font-family: monospace; font-size: 14px; background: #222; padding: 15px; border-radius: 10px; display: inline-block;"></div>
            </div>
        </div>
        
        <!-- Encoder 特征图可视化 -->
        <div class="encoder-viz" id="encoderViz" style="display: none; width: 100%; margin-top: 20px;">
            <div class="pipeline-title">Encoder 特征图</div>
            
            <!-- 步骤1: 12x12 数值网格 -->
            <div class="encoder-step" id="encoderStep1" style="margin-top: 15px;">
                <div class="step-desc" style="text-align: center; margin-bottom: 10px; color: #aaa;">
                    12×12 特征网格 - 按激活强度排序亮起
                </div>
                <div id="featureGrid" style="display: flex; justify-content: center; gap: 2px; flex-wrap: wrap; max-width: 360px; margin: 0 auto;"></div>
            </div>
            
            <!-- 步骤2: 颜色映射 -->
            <div class="encoder-step" id="encoderStep2" style="margin-top: 20px; display: none;">
                <div class="step-desc" style="text-align: center; margin-bottom: 10px; color: #aaa;">
                    颜色映射：蓝(低) → 青 → 绿 → 黄 → 红(高)
                </div>
                <div id="colorGrid" style="display: flex; justify-content: center; gap: 2px; flex-wrap: wrap; max-width: 360px; margin: 0 auto;"></div>
            </div>
            
            <!-- 步骤3: 热力图 -->
            <div class="encoder-step" id="encoderStep3" style="margin-top: 20px; display: none;">
                <div class="step-desc" style="text-align: center; margin-bottom: 10px; color: #aaa;">
                    最终热力图
                </div>
                <div id="heatmapContainer" style="display: flex; flex-direction: column; align-items: center; gap: 10px;"></div>
            </div>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const previewImage = document.getElementById('previewImage');
        const resultZone = document.getElementById('resultZone');
        const resultContent = document.getElementById('resultContent');
        const preprocessViz = document.getElementById('preprocessViz');
        const preprocessSteps = document.getElementById('preprocessSteps');
        const normViz = document.getElementById('normViz');
        const normStep1 = document.getElementById('normStep1');
        const normStep2 = document.getElementById('normStep2');
        const normStep3 = document.getElementById('normStep3');
        const normBefore = document.getElementById('normBefore');
        const normFormula = document.getElementById('normFormula');
        const normAfter = document.getElementById('normAfter');
        const encoderViz = document.getElementById('encoderViz');
        const encoderStep1 = document.getElementById('encoderStep1');
        const encoderStep2 = document.getElementById('encoderStep2');
        const encoderStep3 = document.getElementById('encoderStep3');
        const featureGrid = document.getElementById('featureGrid');
        const colorGrid = document.getElementById('colorGrid');
        const heatmapContainer = document.getElementById('heatmapContainer');
        
        const steps = [
            document.getElementById('step1'),
            document.getElementById('step2'),
            document.getElementById('step3'),
            document.getElementById('step4'),
            document.getElementById('step5')
        ];
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleImage(file);
            }
        });
        
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleImage(file);
            }
        });
        
        function handleImage(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                dropZone.querySelector('.drop-zone-text').style.display = 'none';
                dropZone.querySelector('.drop-zone-hint').style.display = 'none';
                dropZone.classList.add('has-image');
                
                activateStep(0);
                uploadImage(e.target.result, file.name);
            };
            reader.readAsDataURL(file);
        }
        
        function activateStep(index) {
            steps.forEach((step, i) => {
                step.classList.remove('active', 'completed');
            });
            
            for (let i = 0; i <= index; i++) {
                if (i < index) {
                    steps[i].classList.add('completed');
                } else {
                    steps[i].classList.add('active');
                }
            }
        }
        
        function completeStep(index) {
            steps[index].classList.remove('active');
            steps[index].classList.add('completed');
        }
        
        async function uploadImage(imageData, filename) {
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: imageData, filename: filename})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // 显示识别结果区域（先隐藏，等预处理完再显示）
                    resultZone.style.display = 'block';
                    resultContent.innerHTML = '<span class="cursor"></span>';
                    completeStep(4);
                    
                    // 打字机效果
                    const resultText = data.result;
                    let i = 0;
                    const typingSpeed = 30;
                    
                    function typeWriter() {
                        if (i < resultText.length) {
                            resultContent.innerHTML = resultText.substring(0, i + 1) + '<span class="cursor"></span>';
                            i++;
                            setTimeout(typeWriter, typingSpeed);
                        }
                    }
                    
                    function typeWriter() {
                        if (i < resultText.length) {
                            resultContent.innerHTML = resultText.substring(0, i + 1) + '<span class="cursor"></span>';
                            i++;
                            setTimeout(typeWriter, typingSpeed);
                        }
                    }
                    
                    // 显示预处理步骤图片
                    if (data.preprocess_images && data.preprocess_images.length > 0) {
                        preprocessViz.style.display = 'block';
                        preprocessSteps.innerHTML = '';
                        
                        data.preprocess_images.forEach((imgData, index) => {
                            const card = document.createElement('div');
                            card.className = 'preprocess-card';
                            
                            const stepNames = {
                                'original': '原始图像',
                                'grayscale': '灰度转换',
                                'inverted': '自动反转',
                                'cropped': '边缘裁剪',
                                'resized': '缩放',
                                'padded': 'Padding'
                            };
                            
                            card.innerHTML = `
                                <img src="${imgData.image}" alt="${imgData.name}">
                                <div class="step-name">${stepNames[imgData.name] || imgData.name}</div>
                            `;
                            
                            preprocessSteps.appendChild(card);
                            
                            // 动画延迟显示
                            setTimeout(() => {
                                card.classList.add('show');
                            }, index * 300);
                        });
                        
                        });
                        
                        // ========== 完整动画流程 ==========
                        const preprocessTime = data.preprocess_images.length * 300 + 300;  // 预处理完成时间
                        
                        // 1. 预处理完成后，显示归一化
                        setTimeout(() => {
                            if (data.norm_stats) {
                                normViz.style.display = 'block';
                                normStep1.style.display = 'block';
                                normStep2.style.display = 'none';
                                normStep3.style.display = 'none';
                                
                                // 滚动到归一化区域
                                normViz.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                
                                const [before, after] = data.norm_stats;
                                
                                // 显示归一化前
                                normBefore.innerHTML = `min: ${before.min.toFixed(2)}<br>max: ${before.max.toFixed(2)}<br>mean: ${before.mean.toFixed(2)}<br>std: ${before.std.toFixed(2)}`;
                                
                                // 显示计算公式（动画）
                                setTimeout(() => {
                                    normStep2.style.display = 'block';
                                    const formula = `x' = (x - ${before.mean.toFixed(4)}) / ${before.std.toFixed(4)}`;
                                    let i = 0;
                                    normFormula.textContent = '';
                                    function typeFormula() {
                                        if (i < formula.length) {
                                            normFormula.textContent += formula.charAt(i);
                                            i++;
                                            setTimeout(typeFormula, 30);
                                        }
                                    }
                                    typeFormula();
                                }, 1000);
                                
                                // 显示归一化后
                                setTimeout(() => {
                                    normStep3.style.display = 'block';
                                    normAfter.innerHTML = `min: ${after.min.toFixed(4)}<br>max: ${after.max.toFixed(4)}<br>mean: ${after.mean.toFixed(4)}<br>std: ${after.std.toFixed(4)}`;
                                    
                                    // 归一化完成后，显示 Encoder 特征图
                                    setTimeout(() => {
                                        if (data.encoder_features && data.encoder_viz) {
                                            encoderViz.style.display = 'block';
                                            encoderStep1.style.display = 'block';
                                            encoderStep2.style.display = 'none';
                                            
                                            // 滚动到特征图区域
                                            encoderViz.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                            
                                            // 创建 12x12 网格
                                            featureGrid.innerHTML = '';
                                            const features = data.encoder_features;
                                            const flatFeatures = features.flat();
                                            const maxVal = Math.max(...flatFeatures);
                                            const minVal = Math.min(...flatFeatures);
                                            
                                            // 创建144个格子
                                            for (let i = 0; i < 144; i++) {
                                                const cell = document.createElement('div');
                                                cell.className = 'feature-cell';
                                                cell.dataset.value = flatFeatures[i];
                                                cell.dataset.index = i;
                                                featureGrid.appendChild(cell);
                                            }
                                            
                                            // 动画：逐个格子亮起（按激活强度排序）
                                            const sortedIndices = flatFeatures.map((v, i) => ({v, i}))
                                                .sort((a, b) => b.v - a.v)
                                                .map(x => x.i);
                                            
                                            sortedIndices.forEach((idx, rank) => {
                                                setTimeout(() => {
                                                    const cell = featureGrid.children[idx];
                                                    const val = flatFeatures[idx];
                                                    const normalized = (val - minVal) / (maxVal - minVal);
                                                    const gray = Math.round(normalized * 255);
                                                    cell.style.background = `rgb(${gray}, ${gray}, ${gray})`;
                                                    cell.classList.add('active');
                                                    cell.classList.add('fully-active');
                                                    cell.textContent = (val).toFixed(2);
                                                }, rank * 20);
                                            });
                                            
                                            // 步骤1完成后，显示颜色映射步骤
                                            setTimeout(() => {
                                                encoderStep1.style.display = 'none';
                                                encoderStep2.style.display = 'block';
                                                colorGrid.innerHTML = '';
                                                
                                                // 克隆 featureGrid 到 colorGrid
                                                for (let i = 0; i < 144; i++) {
                                                    const cell = document.createElement('div');
                                                    cell.className = 'feature-cell';
                                                    const val = flatFeatures[i];
                                                    cell.dataset.value = val;
                                                    cell.dataset.index = i;
                                                    colorGrid.appendChild(cell);
                                                }
                                                
                                                // 颜色映射函数：蓝->青->绿->黄->红
                                                function getHeatmapColor(v) {
                                                    if (v < 0.25) {
                                                        const t = v / 0.25;
                                                        return [0, Math.round(255 * t), Math.round(255 * (1 - t))];
                                                    } else if (v < 0.5) {
                                                        return [0, 255, 0];
                                                    } else if (v < 0.75) {
                                                        const t = (v - 0.5) / 0.25;
                                                        return [Math.round(255 * t), 255, 0];
                                                    } else {
                                                        const t = (v - 0.75) / 0.25;
                                                        return [255, Math.round(255 * (1 - t)), 0];
                                                    }
                                                }
                                                
                                                // 逐个格子变色
                                                sortedIndices.forEach((idx, rank) => {
                                                    setTimeout(() => {
                                                        const cell = colorGrid.children[idx];
                                                        const val = flatFeatures[idx];
                                                        const normalized = (val - minVal) / (maxVal - minVal);
                                                        const color = getHeatmapColor(normalized);
                                                        cell.style.background = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                                                        cell.classList.add('active');
                                                        cell.classList.add('colored');
                                                        cell.textContent = (val).toFixed(2);
                                                    }, rank * 15);
                                                });
                                                
                                                // 颜色映射完成后显示热力图（逐格扫描）
                                                setTimeout(() => {
                                                    encoderStep2.style.display = 'none';
                                                    encoderStep3.style.display = 'block';
                                                    
                                                    // 创建 12x12 热力图网格
                                                    heatmapContainer.innerHTML = '';
                                                    const heatmapGrid = document.createElement('div');
                                                    heatmapGrid.style.display = 'grid';
                                                    heatmapGrid.style.gridTemplateColumns = 'repeat(12, 16px)';
                                                    heatmapGrid.style.gap = '1px';
                                                    heatmapGrid.style.justifyContent = 'center';
                                                    
                                                    // 颜色映射函数
                                                    function getHeatmapColor2(v) {
                                                        if (v < 0.25) {
                                                            const t = v / 0.25;
                                                            return `rgb(0, ${Math.round(255 * t)}, ${Math.round(255 * (1 - t))})`;
                                                        } else if (v < 0.5) {
                                                            return 'rgb(0, 255, 0)';
                                                        } else if (v < 0.75) {
                                                            const t = (v - 0.5) / 0.25;
                                                            return `rgb(${Math.round(255 * t)}, 255, 0)`;
                                                        } else {
                                                            const t = (v - 0.75) / 0.25;
                                                            return `rgb(255, ${Math.round(255 * (1 - t))}, 0)`;
                                                        }
                                                    }
                                                    
                                                    // 创建格子
                                                    for (let i = 0; i < 144; i++) {
                                                        const cell = document.createElement('div');
                                                        cell.className = 'heatmap-cell';
                                                        heatmapGrid.appendChild(cell);
                                                    }
                                                    
                                                    heatmapContainer.appendChild(heatmapGrid);
                                                    
                                                    // 扫描动画：从左上到右下，逐行扫描
                                                    for (let row = 0; row < 12; row++) {
                                                        for (let col = 0; col < 12; col++) {
                                                            const idx = row * 12 + col;
                                                            const val = flatFeatures[idx];
                                                            const normalized = (val - minVal) / (maxVal - minVal);
                                                            const color = getHeatmapColor2(normalized);
                                                            
                                                            setTimeout(() => {
                                                                const cell = heatmapGrid.children[idx];
                                                                cell.style.background = color;
                                                                cell.classList.add('show');
                                                            }, idx * 30);
                                                        }
                                                    }
                                                    
                                                    // 扫描完成后显示文字
                                                    setTimeout(() => {
                                                        const label = document.createElement('div');
                                                        label.className = 'step-name';
                                                        label.textContent = '特征热力图 - 扫描完成';
                                                        label.style.marginTop = '15px';
                                                        heatmapContainer.appendChild(label);
                                                        
                                                        // 热力图完成后，显示结果
                                                        resultZone.style.display = 'block';
                                                        resultZone.scrollIntoView({ behavior: 'smooth', block: 'end' });
                                                        typeWriter();
                                                    }, 144 * 30 + 500);
                                                    
                                                }, 144 * 15 + 500);
                                                
                                            }, 144 * 20 + 800);
                                        }
                                    }, 1500);  // 归一化完成后等1.5秒再显示Encoder
                                    
                                }, 2000);  // 公式动画2秒后显示归一化后
                                
                            }
                        }, preprocessTime);
                    } else {
                                        const cell = featureGrid.children[idx];
                                        const val = flatFeatures[idx];
                                        const normalized = (val - minVal) / (maxVal - minVal);
                                        const gray = Math.round(normalized * 255);
                                        cell.style.background = `rgb(${gray}, ${gray}, ${gray})`;
                                        cell.classList.add('active');
                                        cell.classList.add('fully-active');
                                        cell.textContent = (val).toFixed(2);
                                    }, rank * 20);
                                });
                                
                                // 步骤1完成后，显示颜色映射步骤
                                setTimeout(() => {
                                    encoderStep1.style.display = 'none';
                                    encoderStep2.style.display = 'block';
                                    colorGrid.innerHTML = '';
                                    
                                    // 克隆 featureGrid 到 colorGrid
                                    for (let i = 0; i < 144; i++) {
                                        const cell = document.createElement('div');
                                        cell.className = 'feature-cell';
                                        const val = flatFeatures[i];
                                        cell.dataset.value = val;
                                        cell.dataset.index = i;
                                        colorGrid.appendChild(cell);
                                    }
                                    
                                    // 颜色映射函数：蓝->青->绿->黄->红
                                    function getHeatmapColor(v) {
                                        // v: 0-1
                                        if (v < 0.25) {
                                            // 蓝 -> 青
                                            const t = v / 0.25;
                                            return [0, Math.round(255 * t), Math.round(255 * (1 - t))];
                                        } else if (v < 0.5) {
                                            // 青 -> 绿
                                            const t = (v - 0.25) / 0.25;
                                            return [0, 255, 0];
                                        } else if (v < 0.75) {
                                            // 绿 -> 黄
                                            const t = (v - 0.5) / 0.25;
                                            return [Math.round(255 * t), 255, 0];
                                        } else {
                                            // 黄 -> 红
                                            const t = (v - 0.75) / 0.25;
                                            return [255, Math.round(255 * (1 - t)), 0];
                                        }
                                    }
                                    
                                    // 逐个格子变色
                                    sortedIndices.forEach((idx, rank) => {
                                        setTimeout(() => {
                                            const cell = colorGrid.children[idx];
                                            const val = flatFeatures[idx];
                                            const normalized = (val - minVal) / (maxVal - minVal);
                                            const color = getHeatmapColor(normalized);
                                            cell.style.background = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                                            cell.classList.add('active');
                                            cell.classList.add('colored');
                                            cell.textContent = (val).toFixed(2);
                                        }, rank * 15);
                                    });
                                    
                                    // 颜色映射完成后显示热力图（逐格扫描）
                                    setTimeout(() => {
                                        encoderStep2.style.display = 'none';
                                        encoderStep3.style.display = 'block';
                                        
                                        // 创建 12x12 热力图网格
                                        heatmapContainer.innerHTML = '';
                                        const heatmapGrid = document.createElement('div');
                                        heatmapGrid.style.display = 'grid';
                                        heatmapGrid.style.gridTemplateColumns = 'repeat(12, 16px)';
                                        heatmapGrid.style.gap = '1px';
                                        heatmapGrid.style.justifyContent = 'center';
                                        
                                        // 颜色映射函数
                                        function getHeatmapColor(v) {
                                            if (v < 0.25) {
                                                const t = v / 0.25;
                                                return `rgb(0, ${Math.round(255 * t)}, ${Math.round(255 * (1 - t))})`;
                                            } else if (v < 0.5) {
                                                return 'rgb(0, 255, 0)';
                                            } else if (v < 0.75) {
                                                const t = (v - 0.5) / 0.25;
                                                return `rgb(${Math.round(255 * t)}, 255, 0)`;
                                            } else {
                                                const t = (v - 0.75) / 0.25;
                                                return `rgb(255, ${Math.round(255 * (1 - t))}, 0)`;
                                            }
                                        }
                                        
                                        // 创建格子
                                        for (let i = 0; i < 144; i++) {
                                            const cell = document.createElement('div');
                                            cell.className = 'heatmap-cell';
                                            const row = Math.floor(i / 12);
                                            const col = i % 12;
                                            cell.dataset.row = row;
                                            cell.dataset.col = col;
                                            heatmapGrid.appendChild(cell);
                                        }
                                        
                                        heatmapContainer.appendChild(heatmapGrid);
                                        
                                        // 扫描动画：从左上到右下，逐行扫描
                                        for (let row = 0; row < 12; row++) {
                                            for (let col = 0; col < 12; col++) {
                                                const idx = row * 12 + col;
                                                const val = flatFeatures[idx];
                                                const normalized = (val - minVal) / (maxVal - minVal);
                                                const color = getHeatmapColor(normalized);
                                                
                                                setTimeout(() => {
                                                    const cell = heatmapGrid.children[idx];
                                                    cell.style.background = color;
                                                    cell.classList.add('show');
                                                }, idx * 30);
                                            }
                                        }
                                        
                                        // 扫描完成后显示文字
                                        setTimeout(() => {
                                            const label = document.createElement('div');
                                            label.className = 'step-name';
                                            label.textContent = '12×12 特征热力图 - 扫描完成';
                                            label.style.marginTop = '15px';
                                            heatmapContainer.appendChild(label);
                                        }, 144 * 30 + 200);
                                        
                                    }, 144 * 15 + 500);
                                    
                                }, 144 * 20 + 800);
                            }
                        }, data.preprocess_images.length * 300 + 500);
                    } else {
                        // 没有预处理图片直接显示结果
                        resultZone.scrollIntoView({ behavior: 'smooth', block: 'end' });
                        typeWriter();
                    }
                } else {
                    alert('处理失败: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('请求失败: ' + error.message);
            }
        }
    </script>
</body>
</html>
'''

# ============ HTTP 服务器 ============
class VisualizerHandler(SimpleHTTPRequestHandler):
    """可视化 HTTP 请求处理器"""
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/process':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body.decode('utf-8'))
                image_data = data.get('image', '')
                filename = data.get('filename', 'image.png')
                
                # 处理图片
                result = process_image(image_data, filename)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def process_image(image_data: str, filename: str) -> dict:
    """处理图片并返回结果"""
    try:
        import numpy as np
        
        # 激活步骤 1: 加载图像
        print(f"[步骤1] 加载图像: {filename}")
        
        # 解码 base64 图片
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 保存临时文件
        temp_path = MODEL_DIR / 'temp_input.png'
        image.save(temp_path)
        
        # 激活步骤 2: 图像预处理
        print("[步骤2] 图像预处理...")
        
        # 初始化模型（如果需要）
        global models, tokenizer, engine
        if models is None:
            print("Initializing models...")
            models = ONNXModels()
            models.load_models()
            tokenizer = Tokenizer()
            tokenizer.load(Config.TOKENIZER_PATH)
            engine = LaTeXInferenceEngine(models, tokenizer)
        
        # 推理
        print("Running inference...")
        result, steps, preprocess_results, encoder_features, norm_stats = engine.infer(str(temp_path))
        
        # 转换预处理结果为图片
        preprocess_images = []
        for name, arr in preprocess_results:
            # 归一化数组转回 uint8
            if arr.dtype != np.uint8:
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            preprocess_images.append({'name': name, 'image': f'data:image/png;base64,{img_b64}'})
        
        # 转换 encoder 特征图为热力图
        encoder_viz = None
        if encoder_features is not None:
            # encoder_features: [batch, 144, 2048] -> 取第一个样本
            features = encoder_features[0]  # [144, 2048]
            # 取所有位置的均值作为特征图
            feat_2d = features.mean(axis=1)  # [144]
            # 保存原始 12x12 数值（用于动画）
            feat_12x12_raw = feat_2d.reshape(12, 12).tolist()
            # 归一化到 0-255
            feat_min = feat_2d.min()
            feat_max = feat_2d.max()
            feat_12x12_norm = ((feat_2d.reshape(12, 12) - feat_min) / (feat_max - feat_min + 1e-8) * 255).astype(np.uint8)
            # 转为灰度图
            feat_img = Image.fromarray(feat_12x12_norm).resize((200, 200), Image.NEAREST)
            # 转为彩色热力图
            feat_np = np.array(feat_img)
            h, w = feat_np.shape
            color = np.zeros((h, w, 3), dtype=np.uint8)
            v = feat_np / 255.0
            # 蓝->青->绿->黄->红
            color[:,:,2] = (1 - v) * 255
            color[:,:,1] = np.where(v > 0.5, (1 - (v - 0.5) * 2) * 255, 255).astype(np.uint8)
            color[:,:,0] = np.where(v > 0.5, (v - 0.5) * 2 * 255, 0).astype(np.uint8)
            # 放大
            color_big = Image.fromarray(color).resize((200, 200), Image.NEAREST)
            buf = io.BytesIO()
            color_big.save(buf, format='PNG')
            encoder_viz = f'data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("utf-8")}'
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            preprocess_images.append({'name': name, 'image': f'data:image/png;base64,{img_b64}'})
        
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()
        
        return {'success': True, 'result': result, 'preprocess_images': preprocess_images, 'encoder_viz': encoder_viz, 'encoder_features': feat_12x12_raw, 'norm_stats': norm_stats}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# 全局变量
models = None
tokenizer = None
engine = None


def run_server(port: int = PORT):
    """运行服务器"""
    # 切换到脚本目录
    os.chdir(MODEL_DIR)
    
    server = HTTPServer(('', port), VisualizerHandler)
    print("LaTeX OCR Visualizer Started")
    print(f"   Open in browser: http://localhost:{port}")
    print(f"   Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.shutdown()


if __name__ == '__main__':
    if not HAS_ORT:
        print("错误: 请先安装 onnxruntime")
        print("   pip install onnxruntime")
        sys.exit(1)
    
    run_server()
