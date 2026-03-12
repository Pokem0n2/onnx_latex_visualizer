"""
LaTeX OCR Visualizer - ONNX 推理可视化工具
==========================================
将图片推理为 LaTeX 公式的全过程可视化
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# 可视化依赖
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("警告: PIL 未安装")

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("错误: onnxruntime 未安装，请运行: pip install onnxruntime")


# ============ 配置 ============
class Config:
    """程序配置"""
    # 模型路径 (脚本所在目录)
    MODEL_DIR = Path(__file__).parent
    ENCODER_PATH = MODEL_DIR / "encoder.onnx"
    DECODER_PATH = MODEL_DIR / "decoder.onnx"
    TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
    CONFIG_PATH = MODEL_DIR / "config.json"

    # 推理参数
    MAX_LENGTH = 1024
    BOS_TOKEN_ID = 0
    EOS_TOKEN_ID = 2
    PAD_TOKEN_ID = 1

    # 可视化配置
    VIS_WIDTH = 800
    VIS_HEIGHT = 600


# ============ 第一步：加载模型 ============
class ONNXModels:
    """ONNX 模型加载器"""

    def __init__(self):
        self.encoder_session = None
        self.decoder_session = None
        self.encoder_input_name = ""
        self.decoder_input_names = {}

    def load_models(self) -> bool:
        """加载 encoder 和 decoder 模型"""
        if not HAS_ORT:
            print("错误: 无法加载 ONNX Runtime")
            return False

        print(f"加载 Encoder: {Config.ENCODER_PATH}")
        self.encoder_session = ort.InferenceSession(
            str(Config.ENCODER_PATH),
            providers=['CPUExecutionProvider']
        )
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        print(f"  Encoder 输入: {self.encoder_input_name}")

        print(f"\n加载 Decoder: {Config.DECODER_PATH}")
        self.decoder_session = ort.InferenceSession(
            str(Config.DECODER_PATH),
            providers=['CPUExecutionProvider']
        )

        # 获取 decoder 输入输出名称
        inputs = self.decoder_session.get_inputs()
        outputs = self.decoder_session.get_outputs()
        self.decoder_input_names = {
            "input_ids": inputs[0].name,
            "encoder_hidden_states": inputs[1].name,
        }

        # KV cache 输入
        for i, inp in enumerate(inputs):
            if inp.name.startswith("past_key_values"):
                self.decoder_input_names[inp.name] = inp.name

        self.decoder_output_names = {
            "logits": outputs[0].name,
        }
        for out in outputs:
            if out.name.startswith("present"):
                self.decoder_output_names[out.name] = out.name

        print(f"  Decoder 输入: {list(self.decoder_input_names.keys())}")
        print(f"  Decoder 输出: {list(self.decoder_output_names.keys())}")

        return True

    def get_encoder_output_shape(self) -> Tuple[int, int, int]:
        """获取 encoder 输出形状 [seq_len, hidden_size]"""
        # 从 config 推断: 12x12 = 144
        return (144, 2048)


# ============ 第二步：Tokenizer ============
class Tokenizer:
    """简易 Tokenizer (基于 tokenizer.json)"""

    # 特殊 token
    SPACING_MACROS = {"\\quad", "\\qquad", "\\,", "\\;", "\\:", "~"}

    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}
        self.bos_token_id = Config.BOS_TOKEN_ID
        self.eos_token_id = Config.EOS_TOKEN_ID
        self.pad_token_id = Config.PAD_TOKEN_ID

    def load(self, tokenizer_path: Path) -> bool:
        """加载 tokenizer"""
        if not tokenizer_path.exists():
            print(f"错误: tokenizer 文件不存在: {tokenizer_path}")
            return False

        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 解析 vocab (model 部分)
        if "model" in data and "vocab" in data["model"]:
            self.vocab = data["model"]["vocab"]
            self.id_to_token = {v: k for k, v in self.vocab.items()}

        print(f"Tokenizer 加载完成, vocab 大小: {len(self.vocab)}")
        return True

    def encode(self, text: str) -> List[int]:
        """文本 -> token ids"""
        # 简单的空格分词
        tokens = text.split()
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # 尝试找到最接近的 token
                ids.append(self.vocab.get("<unk>", 3))
        return ids

    def decode(self, ids: List[int]) -> str:
        """token ids -> 文本"""
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            elif id == self.eos_token_id:
                break
        return " ".join(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """ids -> tokens"""
        return [self.id_to_token.get(i, "<unk>") for i in ids]

    def convert_ids_to_string(self, ids: List[int], remove_spacing: bool = True) -> str:
        """ids -> LaTeX 字符串"""
        tokens = self.convert_ids_to_tokens(ids)
        # 过滤 special token
        special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        parts = []
        for tid, tok in zip(ids, tokens):
            if tid in special_ids:
                continue
            if remove_spacing and tok in self.SPACING_MACROS:
                continue
            parts.append(tok)
        return " ".join(parts)


def format_latex(code: str) -> str:
    """格式化 LaTeX 字符串，清理多余空格"""
    if not code:
        return ""
    tokens = code.strip().split()
    new_tokens = []
    for i in range(len(tokens)):
        token = tokens[i]
        new_tokens.append(token)
        # LaTeX 命令后需要加空格（如果下一个 token 是字母/数字）
        if i < len(tokens) - 1:
            next_token = tokens[i + 1]
            if token.startswith("\\") and len(next_token) > 0 and next_token[0].isalnum():
                new_tokens.append(" ")
    result = "".join(new_tokens)
    return result


# ============ 第三步：图像预处理 ============
class ImagePreprocessor:
    """图像预处理 - UniMerNet 风格"""

    # UniMerNet 参数
    UNIMERNET_MEAN = 0.7931
    UNIMERNET_STD = 0.1738
    TARGET_H = 384
    TARGET_W = 384
    SPACING_MACROS = {"\\quad", "\\qquad", "\\,", "\\;", "\\:", "~"}

    def __init__(self):
        pass

    def load_grayscale_image(self, image_path: str) -> np.ndarray:
        """加载灰度图"""
        if not HAS_PIL:
            return None
        img = Image.open(image_path).convert("L")
        return np.array(img, dtype=np.uint8)

    def invert_if_needed(self, arr: np.ndarray, threshold: int = 200) -> np.ndarray:
        """基于直方图自动判断是否反转"""
        hist = np.bincount(arr.flatten(), minlength=256)
        black_pixels = hist[:threshold].sum()
        white_pixels = hist[threshold:].sum()
        return 255 - arr if black_pixels >= white_pixels else arr

    def crop_margin(self, arr: np.ndarray, threshold: int = 200) -> np.ndarray:
        """裁剪边缘"""
        flat = arr.flatten()
        maxv, minv = int(flat.max()), int(flat.min())
        if maxv == minv:
            return arr
        normalized = ((arr.astype(float) - minv) / (maxv - minv)) * 255.0
        mask = normalized < threshold
        ys, xs = np.where(mask)
        if ys.size == 0:
            return arr
        minY, maxY = int(ys.min()), int(ys.max())
        minX, maxX = int(xs.min()), int(xs.max())
        return arr[minY:maxY + 1, minX:maxX + 1]

    def resize_and_pad(self, arr: np.ndarray) -> np.ndarray:
        """保持宽高比缩放 + 中心 padding"""
        h, w = arr.shape
        minDim = min(self.TARGET_H, self.TARGET_W)
        scale = minDim / min(h, w)
        newW = int(round(w * scale))
        newH = int(round(h * scale))

        if newW > self.TARGET_W or newH > self.TARGET_H:
            ratio = min(self.TARGET_W / newW, self.TARGET_H / newH)
            newW = int(round(newW * ratio))
            newH = int(round(newH * ratio))

        resized = Image.fromarray(arr).resize((newW, newH), Image.LANCZOS)

        # 创建画布并中心粘贴
        canvas = Image.new("L", (self.TARGET_W, self.TARGET_H), color=0)
        padW = (self.TARGET_W - newW) // 2
        padH = (self.TARGET_H - newH) // 2
        canvas.paste(resized, (padW, padH))
        return np.array(canvas, dtype=np.uint8)

    def normalize_image(self, arr: np.ndarray) -> np.ndarray:
        """UniMerNet 归一化"""
        return ((arr.astype(np.float32) / 255.0 - self.UNIMERNET_MEAN) / self.UNIMERNET_STD).astype(np.float32)

    def preprocess_image(self, image_path: str, debug: bool = False) -> Optional[np.ndarray]:
        """完整预处理流程，返回中间结果用于可视化"""
        results = []
        
        # 1. 加载原始图像
        img = Image.open(image_path)
        original = img.copy()
        results.append(('original', np.array(original)))
        
        # 2. 灰度转换
        arr = np.array(img.convert('L'), dtype=np.uint8)
        results.append(('grayscale', arr.copy()))
        
        # 3. 自动反转（黑底白字 vs 白底黑字）
        arr = self.invert_if_needed(arr)
        results.append(('inverted', arr.copy()))

        # 4. 裁剪边缘
        arr = self.crop_margin(arr)
        results.append(('cropped', arr.copy()))

        # 5. 缩放
        h, w = arr.shape
        minDim = min(self.TARGET_H, self.TARGET_W)
        scale = minDim / min(h, w)
        newW = int(round(w * scale))
        newH = int(round(h * scale))
        if newW > self.TARGET_W or newH > self.TARGET_H:
            ratio = min(self.TARGET_W / newW, self.TARGET_H / newH)
            newW = int(round(newW * ratio))
            newH = int(round(newH * ratio))
        resized = Image.fromarray(arr).resize((newW, newH), Image.LANCZOS)
        results.append(('resized', np.array(resized, dtype=np.uint8)))

        # 6. Padding
        canvas = Image.new("L", (self.TARGET_W, self.TARGET_H), color=0)
        padW = (self.TARGET_W - newW) // 2
        padH = (self.TARGET_H - newH) // 2
        canvas.paste(resized, (padW, padH))
        arr = np.array(canvas, dtype=np.uint8)
        results.append(('padded', arr.copy()))

        # 7. 归一化前保存统计数据
        padded_stats = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'max': float(arr.max())
        }
        
        # 7. 归一化
        norm = self.normalize_image(arr)
        
        # 归一化后的统计
        norm_stats = {
            'mean': float(norm.mean()),
            'std': float(norm.std()),
            'min': float(norm.min()),
            'max': float(norm.max())
        }

        # 8. 转为 3 通道 (复制灰度图)
        tensor = norm.reshape(1, 1, self.TARGET_H, self.TARGET_W)
        pixel_values = np.concatenate([tensor, tensor, tensor], axis=1)

        # 返回: pixel_values, results, 归一化前后统计
        return pixel_values, results, (padded_stats, norm_stats)

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载图像（兼容旧接口）"""
        return self.preprocess_image(image_path)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理（兼容旧接口）"""
        # 这个方法现在通过 preprocess_image 调用
        raise NotImplementedError("使用 preprocess_image 代替")

        # 转换为 [C, H, W]
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array


# ============ 第四步：推理引擎 ============
class LaTeXInferenceEngine:
    """LaTeX 推理引擎"""

    def __init__(self, models: ONNXModels, tokenizer: Tokenizer):
        self.models = models
        self.tokenizer = tokenizer
        self.steps = []  # 记录推理步骤用于可视化

    def warmup(self):
        """Warmup 模型"""
        print("  Running warmup...")
        # 随机输入
        pixel_values = np.zeros((1, 3, 384, 384), dtype=np.float32)

        # Encoder
        enc_out = self.models.encoder_session.run(
            None,
            {self.models.encoder_input_name: pixel_values}
        )
        encoder_hidden_states = enc_out[0]

        # 动态获取输入名称
        in_names = [i.name for i in self.models.decoder_session.get_inputs()]
        id_name = next((n for n in ["input_ids", "input", "decoder_input_ids"] if n in in_names), in_names[0])
        enc_name = next((n for n in ["encoder_hidden_states", "encoder_outputs", "encoder_output"] if n in in_names), in_names[1] if len(in_names) > 1 else in_names[0])

        # 初始化 past 并构建 feed
        past = self._init_past_inputs(encoder_hidden_states.shape[1])
        feed = {id_name: np.array([[0]], dtype=np.int64), enc_name: encoder_hidden_states.astype(np.float32)}
        # 添加 past key values
        for k, v in past.items():
            feed[k] = v
        self.models.decoder_session.run(None, feed)
        print("  Warmup done.")

    def _init_past_inputs(self, encoder_seq_len: int, batch: int = 1) -> dict:
        """初始化 KV cache - 动态检测模型参数"""
        past_inputs = {}

        for inp in self.models.decoder_session.get_inputs():
            name = inp.name
            if name.startswith("past_key_values.") or name.startswith("past."):
                shp = inp.shape or []
                num_heads = shp[1] if len(shp) > 1 and isinstance(shp[1], int) else 16
                is_encoder = ".encoder." in name
                past_seq = encoder_seq_len if is_encoder else 0
                head_dim = shp[3] if len(shp) > 3 and isinstance(shp[3], int) else 24
                arr_shape = (batch, int(num_heads), max(0, int(past_seq)), int(head_dim))
                past_inputs[name] = np.zeros(arr_shape, dtype=np.float32)

        # use_cache_branch
        input_names = [i.name for i in self.models.decoder_session.get_inputs()]
        if "use_cache_branch" in input_names:
            past_inputs["use_cache_branch"] = np.array([False], dtype=bool)

        self.past_key_values = past_inputs
        return past_inputs

    def _decoder_step(self, input_ids: np.ndarray, encoder_hidden_states: np.ndarray,
                     use_cache: bool = True) -> Tuple[np.ndarray, dict]:
        """单步 decoder 推理 - 动态输入名检测"""
        # 动态获取输入名称
        in_names = [i.name for i in self.models.decoder_session.get_inputs()]
        id_name = next((n for n in ["input_ids", "input", "decoder_input_ids"] if n in in_names), None)
        enc_name = next((n for n in ["encoder_hidden_states", "encoder_outputs", "encoder_output"] if n in in_names), None)

        if id_name is None or enc_name is None:
            raise RuntimeError("decoder input names not found")

        # 准备输入
        decoder_inputs = {
            id_name: input_ids.astype(np.int64),
            enc_name: encoder_hidden_states.astype(np.float32),
        }

        # 添加 KV cache
        if "use_cache_branch" in in_names:
            decoder_inputs["use_cache_branch"] = np.array([bool(use_cache)], dtype=bool)

        for k, v in self.past_key_values.items():
            if k in in_names and k != "use_cache_branch":
                decoder_inputs[k] = v

        # 推理
        outputs = self.models.decoder_session.run(None, decoder_inputs)

        # 解析输出
        out_names = [o.name for o in self.models.decoder_session.get_outputs()]
        out_map = {name: outputs[i] for i, name in enumerate(out_names)}

        # 获取 logits
        logits = out_map.get("logits", None)
        if logits is None:
            # fallback: 找第一个合适的输出
            logits = next((v for v in out_map.values() if isinstance(v, np.ndarray) and v.dtype == np.float32 and v.ndim >= 2), None)
        if logits is None:
            raise RuntimeError("No logits in decoder outputs")

        # 获取 present map
        present_map = {name: arr.astype(np.float32) for name, arr in out_map.items() if name.startswith("present.") or name.startswith("present")}

        return logits, present_map

    def infer(self, image_path: str) -> Tuple[str, List[Dict], List, np.ndarray, tuple]:
        """推理图片 -> LaTeX 字符串，返回结果、步骤详情、预处理中间结果、encoder特征图、归一化统计"""
        self.steps = []  # 重置步骤记录

        # Step 1: 加载图像 + 预处理
        preprocessor = ImagePreprocessor()
        pixel_values, preprocess_results, norm_stats = preprocessor.preprocess_image(image_path, debug=True)
        if pixel_values is None:
            return "", [], [], None, (None, None)

        self.steps.append({
            "step": 1,
            "name": "加载图像",
            "description": f"从 {image_path} 加载图像",
            "data": {"shape": pixel_values.shape}
        })

        # Step 2: Encoder 推理
        encoder_output = self.models.encoder_session.run(
            None,
            {self.models.encoder_input_name: pixel_values}
        )
        encoder_hidden_states = encoder_output[0]

        self.steps.append({
            "step": 2,
            "name": "Encoder 特征提取",
            "description": "使用 HGNETv2 提取图像特征",
            "data": {"output_shape": encoder_hidden_states.shape}
        })

        # Step 3: Warmup
        if not hasattr(self, 'past_key_values'):
            self.warmup()

        # Step 4: Decoder 自回归生成
        batch = int(encoder_hidden_states.shape[0])
        encoder_seq_len = int(encoder_hidden_states.shape[1])

        # 重新初始化 past
        self._init_past_inputs(encoder_seq_len, batch=batch)

        bos_id = Config.BOS_TOKEN_ID
        eos_id = Config.EOS_TOKEN_ID
        max_len = 100

        input_ids = np.array([[bos_id]], dtype=np.int64)
        generated = [bos_id]
        generation_steps = []
        encoder_past_initialized = False

        for step in range(max_len):
            use_cache = step > 0
            logits, present_map = self._decoder_step(input_ids, encoder_hidden_states, use_cache)

            # 更新 KV cache
            next_past = dict(self.past_key_values)
            for name, arr in present_map.items():
                if ".decoder." in name:
                    mapped = name.replace("present.", "past_key_values.")
                    next_past[mapped] = arr
                elif ".encoder." in name:
                    mapped = name.replace("present.", "past_key_values.")
                    if not encoder_past_initialized:
                        next_past[mapped] = arr

            # 更新 use_cache_branch
            in_names = [i.name for i in self.models.decoder_session.get_inputs()]
            if "use_cache_branch" in in_names:
                next_past["use_cache_branch"] = np.array([True], dtype=bool)

            self.past_key_values = next_past

            # 采样
            if logits.ndim == 3:
                last_logits = logits[:, -1, :]
            elif logits.ndim == 2:
                last_logits = logits
            else:
                last_logits = logits.reshape((1, -1))

            next_token = int(np.argmax(last_logits, axis=-1)[0])
            generated.append(next_token)

            # 记录
            token_str = self.tokenizer.id_to_token.get(next_token, f"<unk:{next_token}>")
            generation_steps.append({
                "step": step + 1,
                "token_id": next_token,
                "token": token_str,
            })

            # 打印
            print(f"  Step {step+1}: {token_str}")

            # 终止
            if next_token == eos_id:
                break

            # 下一轮输入
            input_ids = np.array([[next_token]], dtype=np.int64)
            encoder_past_initialized = True

        # 解码结果
        result = self.tokenizer.convert_ids_to_string(generated)
        result = ' '.join(result.split())

        self.steps.append({
            "step": 3,
            "name": "Decoder 自回归生成",
            "description": f"自回归生成 {len(generation_steps)} 个 token",
            "data": {
                "generated_tokens": len(generation_steps),
                "first_10_tokens": [
                    f"Step {s['step']}: id={s['token_id']}, token={repr(s['token'])}"
                    for s in generation_steps[:10]
                ]
            }
        })

        # 格式化输出
        result = format_latex(result)

        return result, self.steps, preprocess_results, encoder_hidden_states, norm_stats

    def _get_top5(self, logits: np.ndarray) -> List[Tuple[int, float]]:
        """获取 top5 token"""
        top5_idx = np.argsort(logits)[-5:][::-1]
        return [(int(i), float(logits[i])) for i in top5_idx]


# ============ 第五步：可视化 ============
class Visualizer:
    """推理过程可视化"""

    def __init__(self):
        self.steps = []

    def add_step(self, step: Dict):
        """添加推理步骤"""
        self.steps.append(step)

    def render(self) -> str:
        """渲染可视化结果 (文本模式)"""
        output = []
        output.append("=" * 60)
        output.append("推理过程可视化")
        output.append("=" * 60)

        for step in self.steps:
            output.append(f"\n【步骤 {step['step']}】{step['name']}")
            output.append(f"  {step['description']}")
            if "data" in step:
                for k, v in step["data"].items():
                    output.append(f"  {k}: {v}")

        return "\n".join(output)


# ============ 主程序 ============
def main():
    """主函数"""
    print("=" * 60)
    print("LaTeX OCR Visualizer - ONNX 推理可视化")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/4] 加载 ONNX 模型...")
    models = ONNXModels()
    if not models.load_models():
        return

    # 2. 加载 Tokenizer
    print("\n[2/4] 加载 Tokenizer...")
    tokenizer = Tokenizer()
    if not tokenizer.load(Config.TOKENIZER_PATH):
        return

    # 3. 测试图像
    test_image = Config.MODEL_DIR / "22.png"
    if test_image.exists():
        print(f"\n[3/4] 测试推理: {test_image}")

        # 创建推理引擎
        engine = LaTeXInferenceEngine(models, tokenizer)

        # 推理
        result, steps = engine.infer(str(test_image))

        # 可视化
        visualizer = Visualizer()
        for step in steps:
            visualizer.add_step(step)

        print(visualizer.render())

        print(f"\n推理结果: {result}")
    else:
        print(f"\n警告: 测试图像不存在: {test_image}")

    print("\n[4/4] 完成!")


if __name__ == "__main__":
    main()
