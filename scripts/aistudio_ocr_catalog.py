"""Build an OCR/document download decision catalog for AI Studio datasets."""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import math
import re
import shutil
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "aistudio" / ".aistudio-dataset-catalog.json"
STATE_PATH = ROOT / "aistudio" / ".aistudio-batch-state.json"
CACHE_PATH = ROOT / ".aistudio_ocr_cache.json"
OUTPUT_PATH = ROOT / "aistudio数据集.md"
DECISIONS_PATH = ROOT / ".aistudio-ocr-decisions.json"

DETAIL_URL = "https://aistudio.baidu.com/studio/dataset/detail"
SOURCE_URL = "https://aistudio.baidu.com/datasetoverview?orderType=1&task=26"

DETAIL_FIELDS = (
    "datasetId",
    "datasetName",
    "datasetAbs",
    "datasetContent",
    "datasetType",
    "createTime",
    "updateTime",
    "userId",
    "userName",
    "nickname",
    "tags",
    "topicTags",
    "protocolId",
    "protocolName",
    "fileSizeTotalCount",
    "itemVersion",
    "repoName",
    "gitLogin",
    "repoType",
    "selectType",
    "statusCode",
    "useCount",
    "newUseCount",
    "downloadCount",
    "viewCount",
    "collectCount",
    "commentCount",
)

FILE_FIELDS = (
    "fileId",
    "fileName",
    "fileOriginName",
    "fileSize",
    "fileAbs",
    "fileContentType",
    "isPublic",
)

_thread_local = threading.local()


DECISION_LABELS = {
    "A": "A｜优先下载",
    "B": "B｜按需下载",
    "C": "C｜人工复核",
    "D": "D｜不再下载",
}

GENERIC_NAMES = {
    "ocr",
    "ocr数据集",
    "dataset",
    "data",
    "test",
    "测试",
    "训练集",
    "原始数据集",
    "img",
    "image",
    "images",
    "new",
    "work",
    "list",
    "archive",
    "model",
    "det",
    "rec",
    "table",
    "手写",
    "数字识别",
    "total_dataset",
    "test数据集",
    "6663",
    "裴sir",
}

SPAM_TERMS = (
    "回血",
    "炸金花",
    "手机搜狐网",
    "哪里可以开具",
    "你妈妈的稳",
    "手感很舒服",
    "客户福利福利",
    "<img",
    "<script",
)

SYNTHETIC_TERMS = (
    "synthetic",
    "synthtext",
    "mjsynth",
    "synchinese",
    "curvedsyntext",
    "mj+st",
    "mj_st",
    "st_mj",
    "char_enum",
    "合成数据",
    "合成文档",
    "生成数据集",
    "自动生成",
    "text-renderer",
    "脚本生成",
    "程序生成",
    "自己生成的数据",
    "合成的数据集",
    "虚拟身份证",
    "字体生成",
    "模拟身份证",
    "模拟车牌",
    "仿国税网验证码",
    "mosaic 算法生成",
    "tablegeneration",
    "tablegenerator",
    "mj和 st",
    "mj 和 st",
    "master_lmdb",
    "字体数据集",
    "汉字生成器",
    "ocr常用字体",
    "多字体中文ocr数据集",
)

DERIVED_TERMS = (
    "doc3d",
    "裁剪好",
    "个人裁切版",
    "分割后的",
    "拼接",
    "清洗版",
    "切图数据集",
    "拉直印章",
    "处理之后的数据集",
    "抽取的文本",
)

MODEL_TOOL_TERMS = (
    "模型权重",
    "预训练模型",
    "官方手写体识别模型",
    "模型日志",
    "infer模型",
    "pretrain_model",
    "paddleocr-release",
    "paddleocr套件",
    "一键启动包",
    "code&model",
    "源码与安装包",
    "源码",
    "源代码",
    "代码包",
    "静态图模型文件",
    "推理例子",
    "paddleocr dygraph",
    "paddledetection-release",
    "语言包",
    "tessdata",
    "internlm-",
    "qwen2.5",
    "best_model",
    "trocr-chinese-model",
    "公式识别模型",
    "mineru 所需的数据集",
    "ocr识别模型-",
    "算法模型-",
    "paddlehub",
    "warm_cosine",
    "epoch",
)

UNRELATED_TERMS = (
    "火焰检测",
    "人像分割",
    "家庭物品检测",
    "家庭物品识别",
    "水果",
    "螺母",
    "探索武器",
    "服装图片",
    "垃圾图像分类",
    "fish_dataset",
    "airfoil",
    "coco2009",
    "voc2007",
    "图像风格迁移",
    "pix2pix",
    "疫情信息统计",
    "家庭实用菜谱",
    "结构化日志",
    "知识库",
    "视频字幕",
    "手语语言学",
    "二维码检测",
    "打架视频",
    "斗地主",
    "扑克牌",
    "微博",
    "恶意短信",
    "模具融化结晶",
)

TABLE_TERMS = (
    "表格",
    "table",
    "pubtabnet",
    "tsr_",
    "wtw",
    "合并单元格",
)

LAYOUT_TERMS = (
    "版面",
    "版式",
    "layout",
    "文档分析",
    "文档解析",
    "document parsing",
    "mineru",
    "omnidoc",
)

DOC_QA_TERMS = (
    "docvqa",
    "documentvqa",
    "document vqa",
    "文档问答",
    "文档图像的理解和问答",
    "textvqa",
)

KIE_TERMS = (
    "关键信息",
    "信息抽取",
    "kie",
    "wildreceipt",
    "字段识别",
    "结构化识别",
)

DOCUMENT_TERMS = (
    "文档",
    "document",
    "论文",
    "研报",
    "年报",
    "报告",
    "票据",
    "发票",
    "收据",
    "表单",
    "证书",
    "营业执照",
    "身份证",
    "名片",
    "简历",
    "古籍",
    "档案",
    "试卷",
    "作业",
    "书籍",
    "请假条",
    "货单",
    "合同",
    "报表",
    "营养成分表",
    "internet archive",
    "ia_ocr",
    "pixmo-docs",
)

FORMULA_TERMS = (
    "公式",
    "latex",
    "crohme",
    "equation",
    "unimer",
)

HANDWRITING_TERMS = (
    "手写",
    "handwrite",
    "handwriting",
    "hwdb",
    "书法",
    "古文字",
    "甲骨文",
)

SCENE_TERMS = (
    "场景文字",
    "自然场景",
    "街景",
    "scene text",
    "icdar",
    "lsvt",
    "rctw",
    "rects",
    "ctw1500",
    "mtwi",
    "td500",
    "coco-text",
    "hiertext",
    "svt",
    "art dataset",
)

SPECIAL_OCR_TERMS = (
    "车牌",
    "验证码",
    "电表",
    "水表",
    "仪表",
    "液晶",
    "集装箱",
    "vin",
    "印章",
    "银行卡",
    "芯片字符",
    "动车字符",
    "电梯按键",
    "包装生产日期",
    "门牌",
)

OCR_TERMS = (
    "ocr",
    "文字识别",
    "文本识别",
    "文本检测",
    "字符识别",
    "text recognition",
    "text detection",
)

TARGET_EVIDENCE_TERMS = (
    DOCUMENT_TERMS
    + TABLE_TERMS
    + LAYOUT_TERMS
    + DOC_QA_TERMS
    + KIE_TERMS
    + FORMULA_TERMS
    + HANDWRITING_TERMS
    + SCENE_TERMS
    + SPECIAL_OCR_TERMS
    + OCR_TERMS
)

REAL_EVIDENCE_TERMS = (
    "真实",
    "采集",
    "扫描",
    "拍摄",
    "拍照",
    "手机图片",
    "公开数据集",
    "互联网档案",
    "internet archive",
    "来自",
    "竞赛数据",
)

CANONICAL_REAL_TERMS = (
    "wildreceipt",
    "wilddoc",
    "mthv2",
    "docvqa",
    "documentvqa",
    "hwdb",
    "crohme",
    "ccpd",
    "cblprd",
    "baidu-fest",
    "unimer",
    "hme100k",
)


ASSESSMENT_OVERRIDES: dict[int, dict[str, str]] = {
    167941: {
        "decision": "A",
        "reason": "古籍整页图像与说明文件，适合古籍OCR/文档分析",
    },
    174068: {
        "decision": "C",
        "reason": "平台下载量494，文件为营业执照训练集；含文档图片，但来源和标注未说明，先看样例",
    },
    114036: {
        "authenticity": "混合：真实采集+合成/增强",
        "decision": "C",
        "reason": "含网络收集及手工拍摄水表图，也含合成和增强图；复核后只保留真实原图",
    },
    114635: {
        "authenticity": "混合：真实评测+合成训练",
        "decision": "C",
        "reason": "训练集为MJ/ST，但验证包含IIIT5K、SVT、ICDAR等真实评测集；只选真实评测部分",
    },
    124738: {
        "authenticity": "混合：真实评测+合成训练",
        "decision": "C",
        "reason": "同时含MJSynth和ICDAR、IIIT5K、SVT、CUTE等真实评测集；只选evaluation.zip",
    },
    111844: {
        "authenticity": "混合：真实测试待核+合成训练",
        "decision": "C",
        "reason": "合成训练包之外另有Chinese_dataset测试包；需看样例确认测试图真实性",
    },
    111037: {
        "dataset_type": "OCR｜混合文本识别",
        "authenticity": "混合：真实评测+合成训练/模型",
        "granularity": "文本行/字符",
        "decision": "C",
        "reason": "数据包同时含训练数据、真实评测包和模型权重；只抽取真实评测数据",
    },
    138433: {
        "authenticity": "混合：真实场景+合成训练",
        "decision": "C",
        "reason": "详情明确同时包含ICDAR/IIIT5K/COCO-Text真实数据和SynthText类合成数据；只选真实包",
    },
    140025: {
        "authenticity": "混合：真实字符+合成拼接",
        "decision": "C",
        "reason": "含MNIST原始字符及合成数字串；仅在需要字符OCR时抽取非合成部分",
    },
    146978: {
        "authenticity": "混合：真实评测+合成训练",
        "decision": "C",
        "reason": "大部分为SynthText/MJSynth，但另有testing_lmdb和evaluation_spin；先核验并仅保留真实评测包",
    },
    199950: {
        "authenticity": "混合：真实评测+合成训练/模型",
        "decision": "C",
        "reason": "含SVT、ICDAR、IIIT5K、CUTE真实评测包，也混有MJ/ST和模型；只选真实评测文件",
    },
    264704: {
        "authenticity": "混合：真实评测+合成训练",
        "decision": "C",
        "reason": "DTRB包含MJ/ST合成训练集及ICDAR、IIIT、SVT、CUTE真实评测集；解包后筛选",
    },
    114472: {
        "authenticity": "疑似真实评测集",
        "decision": "C",
        "reason": "DTRB验证集通常来自真实基准，但平台说明不足；查看validation.zip样例后决定",
    },
    114473: {
        "authenticity": "疑似真实评测集",
        "decision": "C",
        "reason": "DTRB验证集通常来自真实基准，但平台说明不足；查看validation.zip样例后决定",
    },
    114474: {
        "authenticity": "疑似真实评测集",
        "decision": "C",
        "reason": "evaluation.zip用于六个文字识别基准测试；需核验内部是否为真实评测图",
    },
    40148: {
        "dataset_type": "文档｜KIE/字段抽取",
        "authenticity": "真实脱敏简历，混装模型",
        "granularity": "简历文档/结构化标注",
        "decision": "C",
        "reason": "含脱敏简历、训练标注和样例，也混有2.8GiB模型；复核后只保留数据与标注",
    },
    154392: {
        "dataset_type": "文档｜KIE/字段抽取",
        "authenticity": "业务文档图，混装模型",
        "granularity": "票据/表单整页",
        "decision": "C",
        "reason": "含163张XTOWER文档图及标注，也含LayoutXLM模型；只抽取xtower/tower_min数据包",
    },
    127845: {
        "dataset_type": "OCR｜专用目标",
        "authenticity": "混合：真实拍摄+合成",
        "granularity": "仪表屏幕图",
        "decision": "C",
        "reason": "计量设备屏幕检测集含手机/工业相机拍摄图，也混有合成内容；抽样区分后再用",
    },
    128714: {
        "dataset_type": "OCR｜专用目标",
        "authenticity": "混合：真实拍摄+合成/裁剪",
        "granularity": "仪表文字区域",
        "decision": "C",
        "reason": "含手机/工业相机拍摄的设备屏幕，也含网络、合成及裁剪图；仅筛选真实来源部分",
    },
    178304: {
        "dataset_type": "OCR｜专用目标",
        "authenticity": "混合数据的重复镜像",
        "granularity": "仪表文字区域",
        "decision": "D",
        "reason": "与ID 128714文件大小及描述一致，保留128714复核，本条按重复镜像跳过",
    },
    224789: {
        "dataset_type": "OCR｜混合文本识别",
        "authenticity": "手写与自然场景混合，来源待核",
        "granularity": "文本行/字符",
        "decision": "C",
        "reason": "文件是手写和自然场景识别样本而非模型权重，但来源与组成未说明，先看样例",
    },
    111604: {
        "dataset_type": "OCR｜通用/粒度未知",
        "authenticity": "教程真实照片",
        "granularity": "整图",
        "decision": "C",
        "reason": "包含PaddleHub教程的5张OCR照片，不是PaddleHub模型包；规模很小，仅作样例复核",
    },
    150140: {
        "dataset_type": "文档｜整页OCR",
        "authenticity": "比赛数据，混装代码/权重",
        "granularity": "整页或文字区域",
        "decision": "C",
        "reason": "农业银行OCR竞赛包含train/extend/label数据，也混有代码和权重；只抽取数据文件",
    },
    167570: {
        "dataset_type": "文档｜KIE/字段抽取",
        "authenticity": "官方比赛数据，混装模型",
        "granularity": "票据/表单整页",
        "decision": "C",
        "reason": "给定模板识别比赛包含template_rec、WildReceipt等数据及多个模型；需逐文件筛选",
    },
    295625: {
        "dataset_type": "文档｜整页OCR",
        "authenticity": "测试图，混装模型",
        "granularity": "整图",
        "decision": "C",
        "reason": "包含5张GOT-OCR测试图片及模型权重；只保留图片，样本规模很小",
    },
    191867: {
        "authenticity": "合成/渲染",
        "decision": "C",
        "reason": "含60张合成机动车发票文档图及标注；不进入真实页面白名单，但保留作合成票据OCR辅助数据",
    },
    234905: {
        "dataset_type": "非数据｜模型/工具",
        "authenticity": "未说明",
        "decision": "D",
        "reason": "三个文件均为PaddleOCR代码或预训练/微调模型，不是数据集",
    },
    186255: {
        "authenticity": "混合：合成训练+真实测试",
        "decision": "B",
        "reason": "仅测试部分是真实文档；若下载应只保留真实测试文件",
    },
    218376: {
        "dataset_type": "OCR｜场景文字",
        "granularity": "网络/场景图",
        "decision": "B",
        "reason": "MTWI是真实网络场景OCR基准，但不是原始文档页",
    },
    258969: {
        "dataset_type": "OCR｜场景文字",
        "granularity": "网络/场景图",
        "decision": "B",
        "reason": "MTWI镜像，真实场景OCR但与整页文档目标不一致，并有重复风险",
    },
    182340: {
        "authenticity": "派生/合成混合",
        "decision": "C",
        "reason": "从HWDB字符提取并合成多位数，属于OCR辅助数据；不进入真实页面白名单，按需复核",
    },
    150236: {
        "authenticity": "派生/渲染",
        "decision": "C",
        "reason": "包含LaTeX公式识别图像，虽非真实原始文档页，仍可作为公式OCR辅助数据复核",
    },
    190884: {
        "decision": "C",
        "reason": "ICDAR 2023图像信息抽取任务可能匹配KIE，但平台说明不足，需查看样例与上游条款",
    },
    221058: {
        "decision": "C",
        "reason": "名称指向DocVQA实验子集，但介绍仅写实验用途，需查看样例并核验来源",
    },
    261064: {
        "authenticity": "派生表格图像/公开基准",
        "decision": "B",
        "reason": "PubTabNet适合表格结构识别，但不是原始完整文档页面",
    },
    167546: {
        "authenticity": "派生表格图像/公开基准",
        "decision": "D",
        "reason": "PubTabNet重复镜像，优先保留信息更完整的ID 261064",
    },
    340103: {
        "dataset_type": "OCR｜场景文字/VQA",
        "granularity": "自然图像",
        "decision": "B",
        "reason": "TextVQA基于OpenImages自然图像，不是文档页",
    },
    181757: {
        "dataset_type": "非目标｜元数据冲突",
        "authenticity": "未说明",
        "decision": "D",
        "reason": "名称写表格识别，详情却描述打架视频，元数据明显不可信",
    },
    181721: {
        "authenticity": "官方比赛文档图像",
        "decision": "A",
        "reason": "一万张真实表格训练图及JSON标注，匹配表格检测任务",
    },
    202650: {
        "dataset_type": "OCR｜混合文本识别",
        "granularity": "文本行/字符与街景混合",
        "decision": "B",
        "reason": "包含古籍、生僻字和街景混合裁剪数据，不是原始整页",
    },
    198731: {
        "authenticity": "合成/渲染",
        "decision": "C",
        "reason": "由TableGeneration生成的长表格图像，属于合成表格识别数据；保留在C类按辅助用途复核",
    },
    350872: {
        "dataset_type": "非目标｜OCR后校正",
        "granularity": "文本/方法集合",
        "decision": "D",
        "reason": "面向OCR后文本纠错，未证明包含真实原始文档图像",
    },
    106617: {
        "dataset_type": "OCR｜验证码",
        "granularity": "验证码裁剪",
        "decision": "D",
        "reason": "验证码绕过示例，与真实文档OCR目标无关",
    },
    312435: {
        "dataset_type": "非数据｜模型/环境包",
        "decision": "D",
        "reason": "详情写明为MinerU模型，文件主要是模型与whl环境包",
    },
    369617: {
        "decision": "D",
        "reason": "仓库无公开文件且无介绍，不能作为可下载数据源",
    },
    343492: {
        "dataset_type": "文档｜VQA/理解",
        "authenticity": "合成文档图/合成问答",
        "granularity": "合成图表/表格/图示/文档",
        "decision": "C",
        "reason": "含代码渲染的图表、表格、图示和文档图片及合成问答；不进真实页面白名单，但可作合成文档理解辅助数据",
    },
    186036: {
        "authenticity": "真实公开公司年报",
        "decision": "A",
        "reason": "包含真实上市公司年度报告，适合作为原始PDF/文档页面语料",
    },
    295072: {
        "decision": "A",
        "reason": "上传者明确说明来自官方论文数据集，适合版面分析；许可仍需核验",
    },
    352473: {
        "authenticity": "公开文档VQA基准",
        "decision": "A",
        "reason": "标准文档图像问答数据，匹配文档理解目标",
    },
    343618: {
        "authenticity": "公开文档VQA基准子集",
        "decision": "A",
        "reason": "DocVQA测试子集，适合文档理解评测",
    },
    343616: {
        "authenticity": "混合：真实报告页+合成查询",
        "decision": "B",
        "reason": "含1,538张真实ESG报告页，但检索查询由模型生成且原文版权需复核",
    },
    165369: {
        "authenticity": "真实古籍图像",
        "decision": "A",
        "reason": "包含古籍图像、文字位置和Unicode转写，匹配古籍整页OCR",
    },
    208531: {
        "authenticity": "混合：真实拍摄+合成",
        "decision": "B",
        "reason": "包含两百多张手机拍摄请假条，也混入两百多张合成页面",
    },
    205685: {
        "authenticity": "官方比赛文档图像",
        "decision": "A",
        "reason": "百度网盘版式分析比赛训练/测试图像，适合文档版面任务",
    },
    221124: {
        "decision": "D",
        "reason": "与ID 205685文件大小和划分一致但说明更少，按重复镜像跳过",
    },
    133260: {
        "decision": "C",
        "reason": "表格结构比赛训练包，但来源、标注格式和页面真实性未充分说明",
    },
    133551: {
        "decision": "C",
        "reason": "表格结构比赛A榜包，缺少来源、标注和样例说明",
    },
    289002: {
        "decision": "C",
        "reason": "名称指向营养成分表照片，但页面未说明采集来源和标注内容",
    },
    127676: {
        "decision": "C",
        "reason": "可能含古籍页面切割数据，但来源和标注格式仍需查看样例",
    },
}


@dataclass(frozen=True)
class Assessment:
    dataset_id: int
    rank: int
    name: str
    dataset_type: str
    authenticity: str
    granularity: str
    annotation: str
    evidence: str
    decision: str
    reason: str
    risks: tuple[str, ...]
    quantity: str
    size_bytes: int
    local_status: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "aistudio-ocr-catalog/1.0",
            }
        )
        _thread_local.session = session
    return session


def compact_detail(result: dict[str, Any]) -> dict[str, Any]:
    detail = {key: result.get(key) for key in DETAIL_FIELDS}
    files = []
    for raw_file in result.get("fileList") or []:
        if not isinstance(raw_file, dict):
            continue
        files.append({key: raw_file.get(key) for key in FILE_FIELDS})
    detail["fileList"] = files
    return detail


def fetch_detail(dataset_id: int, attempts: int = 6) -> dict[str, Any]:
    session = get_session()
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.post(
                DETAIL_URL,
                data={"datasetId": dataset_id},
                timeout=(20, 60),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("detail endpoint returned a non-object")
            if str(payload.get("errorCode", 0)) != "0":
                raise RuntimeError(
                    str(payload.get("errorMsg") or payload.get("errorCode"))
                )
            result = payload.get("result", payload)
            if not isinstance(result, dict):
                raise RuntimeError("detail endpoint result is not an object")
            if int(result.get("datasetId") or 0) != dataset_id:
                raise RuntimeError(
                    f"detail endpoint returned dataset {result.get('datasetId')}"
                )
            return compact_detail(result)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(12.0, 1.5 * (2**attempt)))
    raise RuntimeError(f"dataset {dataset_id}: {last_error}") from last_error


def load_catalog() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog = load_json(CATALOG_PATH)
    datasets = catalog.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise RuntimeError(f"No datasets in {CATALOG_PATH}")
    if len(datasets) != int(catalog.get("unique_count") or len(datasets)):
        raise RuntimeError("Catalog dataset count does not match unique_count")
    return catalog, datasets


def load_cached_details() -> dict[int, dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    payload = load_json(CACHE_PATH)
    details = payload.get("details")
    if not isinstance(details, dict):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for key, value in details.items():
        if isinstance(value, dict):
            result[int(key)] = value
    return result


def cache_payload(
    catalog: dict[str, Any], details: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "version": 1,
        "source_url": SOURCE_URL,
        "catalog_fetched_at": catalog.get("fetched_at"),
        "updated_at": utc_now(),
        "expected_count": int(catalog.get("unique_count") or 0),
        "details": {str(key): details[key] for key in sorted(details)},
    }


def collect(refresh: bool, workers: int) -> dict[str, Any]:
    catalog, datasets = load_catalog()
    details = {} if refresh else load_cached_details()
    wanted_ids = {int(item["dataset_id"]) for item in datasets}
    details = {key: value for key, value in details.items() if key in wanted_ids}
    pending = [dataset_id for dataset_id in wanted_ids if dataset_id not in details]

    print(
        f"Catalog: {len(datasets)}; cached: {len(details)}; pending: {len(pending)}",
        flush=True,
    )
    completed_since_save = 0
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_detail, dataset_id): dataset_id
            for dataset_id in pending
        }
        for future in concurrent.futures.as_completed(futures):
            dataset_id = futures[future]
            try:
                details[dataset_id] = future.result()
                completed_since_save += 1
                print(
                    f"[{len(details):3d}/{len(datasets)}] {dataset_id} "
                    f"{details[dataset_id].get('datasetName') or ''}",
                    flush=True,
                )
            except Exception as exc:  # keep successful requests resumable
                failures.append(str(exc))
                print(f"[ERROR] {exc}", flush=True)
            if completed_since_save >= 25:
                atomic_write_json(CACHE_PATH, cache_payload(catalog, details))
                completed_since_save = 0

    atomic_write_json(CACHE_PATH, cache_payload(catalog, details))
    missing = sorted(wanted_ids - set(details))
    if failures or missing:
        raise RuntimeError(
            f"Detail collection incomplete: {len(missing)} missing; "
            f"failures={failures[:5]}"
        )
    return cache_payload(catalog, details)


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in terms)


def normalize_markup(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"(?is)<(?:script|style).*?>.*?</(?:script|style)>", " ", text)
    text = re.sub(r"(?i)<(?:br|/p|/div|/li|/tr)\s*/?>", "；", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[[^]]*]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[；;]{2,}", "；", text)
    return text.strip(" ；")


def record_text(record: dict[str, Any]) -> str:
    file_parts = []
    for item in record.get("fileList") or []:
        file_parts.extend(
            [str(item.get("fileOriginName") or item.get("fileName") or ""),
             str(item.get("fileAbs") or "")]
        )
    parts = [
        record.get("datasetName"),
        record.get("datasetAbs"),
        record.get("datasetContent"),
        " ".join(str(tag) for tag in record.get("tags") or []),
        " ".join(file_parts),
    ]
    return normalize_markup(" ".join(str(part) for part in parts if part))


def target_evidence_text(record: dict[str, Any]) -> str:
    """Return target evidence without the platform's noisy category tags."""
    file_parts = []
    for item in record.get("fileList") or []:
        file_parts.extend(
            [
                str(item.get("fileOriginName") or item.get("fileName") or ""),
                str(item.get("fileAbs") or ""),
            ]
        )
    parts = [
        record.get("datasetName"),
        record.get("datasetAbs"),
        record.get("datasetContent"),
        " ".join(file_parts),
    ]
    return normalize_markup(" ".join(str(part) for part in parts if part))


def concise_text(value: str, limit: int = 260) -> str:
    text = normalize_markup(value)
    if len(text) <= limit:
        return text
    cut = max(
        text.rfind("。", 0, limit),
        text.rfind("；", 0, limit),
        text.rfind(". ", 0, limit),
    )
    if cut >= limit // 2:
        return text[: cut + 1]
    return text[:limit].rstrip(" ，,；;。") + "…"


def human_size(size: int | float | None) -> str:
    if size is None:
        return "平台未提供"
    value = float(size)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value):,} {unit}"
    return f"{value:,.2f} {unit}"


def quantity(record: dict[str, Any]) -> str:
    text = " ".join(
        [
            normalize_markup(record.get("datasetAbs")),
            normalize_markup(record.get("datasetContent")),
        ]
    )
    numeric = (
        r"(?:(?:\d{1,3}(?:[,，]\d{3})+|\d+(?:\.\d+)?)\s*(?:万|亿|[wWkK])?"
        r"|[零〇一二两三四五六七八九十百千万亿]+)"
    )
    chinese_unit = (
        r"(?:张|页|份|条|个文档|篇文档|个页面|篇|组|套|类|幅|字符|样本|"
        r"图片|图像|文件|道题|个问题|条问答|个问答|问答对)"
    )
    english_unit = (
        r"(?:(?:PDF|document|image|unit)\s+)?"
        r"(?:pages?|documents?|docs?|images?|questions?|samples?|files?|"
        r"test\s+cases?|cases?|rows?|records?|instances?)\b"
    )
    pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:约|共|包含|总计|合计|共有|一共|总共|近|approximately|about|~)?\s*"
        + numeric
        + r"\s*(?:余|多|左右|以上|余个|多个)?\s*(?:"
        + chinese_unit
        + r"|"
        + english_unit
        + r")",
        flags=re.IGNORECASE,
    )
    matches: list[str] = []
    for match in pattern.finditer(text):
        value = re.sub(r"\s+", " ", match.group(0)).strip().replace("，", ",")
        if not re.search(
            r"\b(?:pages?|documents?|docs?|images?|questions?|samples?|files?|"
            r"test\s+cases?|cases?|rows?|records?|instances?)\b",
            value,
            flags=re.IGNORECASE,
        ):
            value = value.replace(" ", "")
        if value not in matches:
            matches.append(value)
        if len(matches) == 2:
            break
    if matches:
        return " / ".join(matches)
    file_count = len(record.get("fileList") or [])
    return f"平台未说明（{file_count}个文件）" if file_count else "平台未说明"


def file_role(item: dict[str, Any]) -> str:
    name = str(item.get("fileOriginName") or item.get("fileName") or "")
    description = str(item.get("fileAbs") or "")
    text = f"{name} {description}".casefold()
    suffix = Path(name).suffix.casefold()
    if contains_any(
        text,
        (
            "checkpoint",
            "checkpoints",
            "模型",
            "权重",
            "pretrain",
            "best_model",
            "infer",
            "onnx",
            "weight",
            "weights",
            "pdparams",
            "pdopt",
            "internlm",
            "qwen",
        ),
    ) or suffix in {
        ".pth",
        ".pt",
        ".ckpt",
        ".onnx",
        ".pdparams",
        ".pdopt",
        ".safetensors",
    } or re.search(r"(?:^|[_-])(?:ch_)?pp-?ocr.*(?:det|rec|student)", text):
        return "model"
    if suffix in {".py", ".ipynb", ".exe", ".whl"} or contains_any(
        text, ("代码", "脚本", "安装包", "运行环境", "日志")
    ):
        return "tool"
    if suffix in {".json", ".jsonl", ".xml", ".csv", ".tsv", ".txt"} or contains_any(
        text, ("标签", "标注", "label", "annotation", "ground truth", "真值")
    ):
        return "annotation"
    if suffix in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".pdf",
        ".lmdb",
    } or contains_any(text, ("训练数据", "测试数据", "图片", "图像", "数据集")):
        return "data"
    return "archive_or_unknown"


def file_role_summary(record: dict[str, Any]) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    for item in record.get("fileList") or []:
        role = file_role(item)
        counts[role] += 1
        sizes[role] += int(item.get("fileSize") or 0)
    return counts, sizes


def file_summary(record: dict[str, Any], limit: int = 7) -> str:
    files = list(record.get("fileList") or [])
    if not files:
        return "无公开文件"
    ordered = sorted(files, key=lambda item: int(item.get("fileSize") or 0), reverse=True)
    shown = ordered[:limit]
    values = []
    for item in shown:
        name = str(item.get("fileOriginName") or item.get("fileName") or "未命名")
        values.append(f"{name} ({human_size(int(item.get('fileSize') or 0))})")
    if len(files) > limit:
        values.append(f"另{len(files) - limit}个")
    return "；".join(values)


def description_quality(record: dict[str, Any]) -> str:
    abstract = normalize_markup(record.get("datasetAbs"))
    content = normalize_markup(record.get("datasetContent"))
    file_descriptions = " ".join(
        normalize_markup(item.get("fileAbs")) for item in record.get("fileList") or []
    ).strip()
    score = 0
    if len(abstract) >= 20:
        score += 1
    if len(abstract) >= 60:
        score += 1
    if len(content) >= 80:
        score += 2
    elif len(content) >= 25:
        score += 1
    if len(file_descriptions) >= 12:
        score += 1
    if re.search(r"\d+\s*(?:万|张|页|份|条|个|类)", abstract + content):
        score += 1
    if contains_any(abstract + content, ("标注", "来源", "格式", "训练集", "测试集")):
        score += 1
    if score >= 4:
        return "高"
    if score >= 2:
        return "中"
    return "低"


def looks_gibberish(record: dict[str, Any], text: str) -> bool:
    del record
    return contains_any(text, SPAM_TERMS)


def model_or_tool_only(record: dict[str, Any], text: str) -> bool:
    name = normalize_markup(record.get("datasetName"))
    counts, sizes = file_role_summary(record)
    total = sum(sizes.values())
    model_share = sizes["model"] / total if total else 0.0
    tool_share = sizes["tool"] / total if total else 0.0
    explicit = contains_any(text, MODEL_TOOL_TERMS)
    name_only = bool(
        re.search(r"(?:模型|权重|日志|代码|套件|语言包)$", name, flags=re.IGNORECASE)
    ) and "挑战赛" not in name
    data_evidence = contains_any(
        normalize_markup(record.get("datasetAbs")) + " " + normalize_markup(record.get("datasetContent")),
        ("训练数据", "测试数据", "图像数据", "图片数据", "标注数据", "数据集包含"),
    )
    return (
        model_share >= 0.8
        or tool_share >= 0.8
        or ((explicit or name_only) and not data_evidence)
    )


def task_type(record: dict[str, Any], text: str) -> str:
    if model_or_tool_only(record, text):
        return "非数据｜模型/工具"
    if contains_any(text, UNRELATED_TERMS):
        return "非目标｜普通视觉/文本"
    if contains_any(text, TABLE_TERMS):
        return "文档｜表格/结构"
    if contains_any(text, DOC_QA_TERMS):
        return "文档｜VQA/理解"
    if contains_any(text, KIE_TERMS):
        return "文档｜KIE/字段抽取"
    if contains_any(text, SCENE_TERMS):
        return "OCR｜场景文字"
    if contains_any(text, LAYOUT_TERMS):
        return "文档｜版面/解析"
    if contains_any(text, FORMULA_TERMS):
        return "文档｜公式识别"
    if contains_any(text, DOCUMENT_TERMS):
        if contains_any(text, HANDWRITING_TERMS):
            return "文档｜古籍/手写页"
        return "文档｜整页OCR"
    if contains_any(text, HANDWRITING_TERMS):
        return "OCR｜手写/字符"
    if contains_any(text, SPECIAL_OCR_TERMS):
        return "OCR｜专用目标"
    if contains_any(text, OCR_TERMS):
        return "OCR｜通用/粒度未知"
    return "非目标｜无关或信息不足"


def authenticity(record: dict[str, Any], text: str, dataset_type: str) -> str:
    if contains_any(text, SYNTHETIC_TERMS):
        return "合成/渲染"
    if contains_any(text, DERIVED_TERMS):
        return "派生/裁剪/变形"
    if contains_any(text, CANONICAL_REAL_TERMS):
        return "公开基准/真实场景"
    if contains_any(text, REAL_EVIDENCE_TERMS):
        return "真实采集/公开来源"
    if re.search(
        r"(?:来源|地址|github|tianchi|competition|大赛|竞赛).{0,80}https?://",
        text,
        flags=re.IGNORECASE,
    ):
        return "真实采集/公开来源"
    if contains_any(text, SCENE_TERMS) and not dataset_type.startswith("非"):
        return "公开基准/真实场景"
    if dataset_type.startswith("文档｜") and description_quality(record) != "低":
        return "疑似真实，来源待核"
    return "未说明"


def granularity(dataset_type: str, text: str) -> str:
    if dataset_type in {"文档｜版面/解析", "文档｜VQA/理解", "文档｜整页OCR"}:
        return "整页/PDF"
    if dataset_type == "文档｜表格/结构":
        return "整页或表格区域"
    if dataset_type == "文档｜KIE/字段抽取":
        return "票据/表单整页"
    if dataset_type == "文档｜古籍/手写页":
        return "整页或文本行"
    if dataset_type == "文档｜公式识别":
        return "公式区域/行"
    if dataset_type == "OCR｜场景文字":
        return "场景图/文字区域"
    if dataset_type == "OCR｜手写/字符":
        return "字符/文本行"
    if dataset_type == "OCR｜专用目标":
        return "专用目标区域"
    if contains_any(text, ("文本行", "裁剪", "字符", "rec_data", "lmdb")):
        return "文本行/字符"
    return "未说明"


def annotation_summary(text: str) -> str:
    labels = []
    if contains_any(text, ("版面", "layout", "阅读顺序", "区域类别")):
        labels.append("版面区域")
    if contains_any(text, ("表格结构", "html", "单元格", "行列")):
        labels.append("表格结构/HTML")
    if contains_any(text, ("关键信息", "信息抽取", "实体", "字段")):
        labels.append("KIE字段/实体")
    if contains_any(text, ("问答", "question", "answer", "vqa")):
        labels.append("问答")
    if contains_any(text, ("四边形", "多边形", "坐标", "bbox", "检测框", "文本框", "标记文本")):
        labels.append("文字框/坐标")
    if contains_any(
        text,
        (
            "转写",
            "誊录",
            "听录",
            "抄录",
            "transcription",
            "transcribed",
            "识别标签",
            "文字标签",
            "文本标签",
            "label",
            "标注文本",
        ),
    ):
        labels.append("文本转写")
    if contains_any(text, ("latex", "公式")):
        labels.append("LaTeX/公式")
    if not labels and contains_any(text, ("标注", "annotation", "ground truth", "真值")):
        labels.append("有标注，格式未说明")
    return "、".join(dict.fromkeys(labels)) or "未说明"


def local_status_for(state_item: dict[str, Any] | None) -> str:
    if not state_item:
        return "未开始"
    status = str(state_item.get("status") or "")
    if status == "completed":
        return "已完整下载"
    if status == "failed":
        return "失败/保留断点"
    if status in {"running", "interrupted"}:
        return "中断/保留断点"
    if status.startswith("skipped"):
        return f"已跳过（{status.removeprefix('skipped_')}）"
    return status or "未开始"


def assess_record(record: dict[str, Any]) -> Assessment:
    dataset_id = int(record["datasetId"])
    text = record_text(record)
    quality = description_quality(record)
    kind = task_type(record, text)
    auth = authenticity(record, text, kind)
    grain = granularity(kind, text)
    annotation = annotation_summary(text)
    risks: list[str] = []
    counts, sizes = file_role_summary(record)
    total_size = int(record.get("fileSizeTotalCount") or sum(sizes.values()))

    if quality == "低":
        risks.append("描述/标注信息不足")
    if total_size >= 50 * 2**30:
        risks.append("超大数据集")
    if sizes["model"]:
        risks.append(f"含模型/权重约{human_size(sizes['model'])}")
    if sizes["tool"]:
        risks.append("含代码/工具文件")
    license_name = str(record.get("protocolName") or "未声明")
    if "NC" in license_name or "ND" in license_name:
        risks.append("许可限制商用或演绎")
    if license_name in {"其他", "未声明"}:
        risks.append("许可需人工核验")
    name = normalize_markup(record.get("datasetName"))
    if name.startswith("【转】") or name.startswith("转"):
        risks.append("转载/重复风险")
    if contains_any(text, DERIVED_TERMS):
        risks.append("不是原始页面")
    duplicate_ids = record.get("_duplicate_ids") or []
    if duplicate_ids:
        peers = "、".join(str(value) for value in duplicate_ids[:5])
        risks.append(f"同文件大小签名疑似重复（关联ID：{peers}）")

    if not record.get("fileList") or total_size <= 0:
        decision, reason = "D", "没有公开文件或文件大小为零，无法作为可下载数据源"
    elif looks_gibberish(record, text):
        decision, reason = "D", "描述含明确广告、垃圾或恶意内容，不作为数据源"
    elif kind.startswith("非数据｜"):
        decision, reason = "D", "以模型、权重、代码或工具为主，不是目标数据"
    elif kind.startswith("非目标｜"):
        decision, reason = "D", "与文档/OCR目标无关或没有足够相关证据"
    elif auth in {"合成/渲染", "派生/裁剪/变形"}:
        decision, reason = (
            "C",
            "与OCR/文档任务相关，但属于合成、渲染或派生数据；不进入真实原始页面白名单，按辅助用途复核",
        )
    elif name.startswith("【转】"):
        decision, reason = "D", "转载镜像，优先保留来源更清楚的版本"
    elif quality == "低":
        name_evidence = (
            normalize_markup(record.get("datasetName"))
            + " "
            + normalize_markup(record.get("datasetAbs"))
        )
        if auth == "公开基准/真实场景" and contains_any(
            name_evidence,
            CANONICAL_REAL_TERMS + SCENE_TERMS,
        ):
            decision, reason = "B", "公开基准镜像，但该上传条目的说明不足，按需使用"
        elif contains_any(target_evidence_text(record), TARGET_EVIDENCE_TERMS) and (
            sizes["data"] > 0 or sizes["archive_or_unknown"] > 0
        ):
            decision, reason = (
                "C",
                "名称、介绍或文件名明确指向OCR/文档数据，但真实性、粒度或标注不足，先看样例",
            )
        else:
            decision, reason = (
                "C",
                "公开信息不足，无法确认是否含目标数据，也没有明确排除证据，保留人工复核",
            )
    elif kind in {
        "文档｜版面/解析",
        "文档｜VQA/理解",
        "文档｜KIE/字段抽取",
        "文档｜整页OCR",
        "文档｜表格/结构",
    } and grain in {"整页/PDF", "整页或表格区域", "票据/表单整页"}:
        if auth in {"真实采集/公开来源", "公开基准/真实场景"}:
            decision, reason = "A", "真实/疑似真实文档，适合整页OCR或文档解析"
        else:
            decision, reason = "C", "任务匹配，但原始页面来源仍需人工核验"
    elif kind in {"文档｜古籍/手写页", "文档｜公式识别"}:
        if auth in {"真实采集/公开来源", "公开基准/真实场景"}:
            decision, reason = "B", "属于文档OCR辅助任务，按古籍/手写/公式需求选择"
        else:
            decision, reason = "C", "任务相关，但真实来源和页面粒度仍需人工核验"
    elif kind.startswith("OCR｜"):
        if auth in {"真实采集/公开来源", "公开基准/真实场景"}:
            decision, reason = "B", "是真实OCR数据，但不是核心整页文档"
        else:
            decision, reason = "C", "OCR相关但真实性、粒度或来源仍需核实"
    else:
        decision, reason = "D", "不满足当前下载口径"

    override = ASSESSMENT_OVERRIDES.get(dataset_id, {})
    kind = override.get("dataset_type", kind)
    auth = override.get("authenticity", auth)
    grain = override.get("granularity", grain)
    decision = override.get("decision", decision)
    reason = override.get("reason", reason)

    return Assessment(
        dataset_id=dataset_id,
        rank=int(record["_rank"]),
        name=name,
        dataset_type=kind,
        authenticity=auth,
        granularity=grain,
        annotation=annotation,
        evidence=quality,
        decision=decision,
        reason=reason,
        risks=tuple(risks),
        quantity=quantity(record),
        size_bytes=total_size,
        local_status=local_status_for(record.get("_state")),
    )


def build_records() -> list[dict[str, Any]]:
    _, catalog_datasets = load_catalog()
    cache = load_json(CACHE_PATH)
    details = cache.get("details")
    if not isinstance(details, dict) or len(details) != len(catalog_datasets):
        raise RuntimeError("Detail cache is incomplete; run --collect-only first")
    state_items: dict[str, Any] = {}
    if STATE_PATH.exists():
        state = load_json(STATE_PATH)
        if isinstance(state.get("items"), dict):
            state_items = state["items"]

    records = []
    for rank, catalog_item in enumerate(catalog_datasets, 1):
        dataset_id = int(catalog_item["dataset_id"])
        detail = details.get(str(dataset_id))
        if not isinstance(detail, dict):
            raise RuntimeError(f"Missing cached detail for {dataset_id}")
        record = dict(detail)
        record["_rank"] = rank
        record["_repo_id"] = catalog_item.get("repo_id")
        record["_state"] = state_items.get(str(dataset_id))
        records.append(record)

    signature_groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for record in records:
        sizes = tuple(
            sorted(
                int(item.get("fileSize") or 0)
                for item in record.get("fileList") or []
                if int(item.get("fileSize") or 0) > 0
            )
        )
        if sizes and sum(sizes) >= 1024 * 1024:
            signature_groups.setdefault(sizes, []).append(record)
    for group in signature_groups.values():
        if len(group) < 2:
            continue
        quality_score = {"低": 0, "中": 1, "高": 2}
        primary = max(
            group,
            key=lambda item: (
                item.get("_state", {}).get("status") == "completed"
                if isinstance(item.get("_state"), dict)
                else False,
                quality_score[description_quality(item)],
                int(item.get("newUseCount") or item.get("useCount") or 0),
                -int(item["_rank"]),
            ),
        )
        ids = [int(item["datasetId"]) for item in group]
        for record in group:
            record["_duplicate_ids"] = [
                value for value in ids if value != int(record["datasetId"])
            ]
            record["_duplicate_primary_id"] = int(primary["datasetId"])
    return records


def markdown_cell(value: Any) -> str:
    return (
        str(value)
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def intro_and_annotation(record: dict[str, Any], assessment: Assessment) -> str:
    abstract = normalize_markup(record.get("datasetAbs"))
    content = normalize_markup(record.get("datasetContent"))
    parts = []
    if abstract:
        parts.append(abstract)
    if content and content not in abstract and abstract not in content:
        parts.append(content)
    intro = concise_text("；".join(parts), 320) or "平台未提供有效介绍。"
    return f"{intro}<br>**标注：**{assessment.annotation}"


def dataset_display(record: dict[str, Any], assessment: Assessment) -> str:
    dataset_id = assessment.dataset_id
    link = f"https://aistudio.baidu.com/datasetdetail/{dataset_id}"
    uploader = normalize_markup(record.get("nickname") or record.get("userName")) or "未知"
    created = str(record.get("createTime") or "未知")[:10]
    updated = str(record.get("updateTime") or "未知")[:10]
    repo = f"<br>Repo: {record['_repo_id']}" if record.get("_repo_id") else ""
    return (
        f"[{markdown_cell(assessment.name)}]({link})<br>"
        f"ID `{dataset_id}`；上传者：{markdown_cell(uploader)}{repo}<br>"
        f"创建 {created}；更新 {updated}"
    )


def type_display(record: dict[str, Any], assessment: Assessment) -> str:
    tags = "、".join(str(tag) for tag in record.get("tags") or []) or "无"
    return f"{assessment.dataset_type}<br>平台标签：{markdown_cell(tags)}"


def access_display(record: dict[str, Any]) -> str:
    license_name = str(record.get("protocolName") or "未声明")
    generation = "Git仓库型" if record.get("_repo_id") else "旧式文件型"
    return f"{markdown_cell(license_name)}；公开；{generation}<br>平台声明，来源条款需复核"


def popularity_display(record: dict[str, Any]) -> str:
    return (
        f"使用 {int(record.get('newUseCount') or record.get('useCount') or 0):,}<br>"
        f"下载 {int(record.get('downloadCount') or 0):,}；"
        f"浏览 {int(record.get('viewCount') or 0):,}<br>"
        f"收藏 {int(record.get('collectCount') or 0):,}；"
        f"评论 {int(record.get('commentCount') or 0):,}"
    )


def risk_display(assessment: Assessment) -> str:
    risks = "；".join(assessment.risks) if assessment.risks else "未发现明显元数据风险"
    return f"证据：{assessment.evidence}<br>{markdown_cell(risks)}"


def render_row(record: dict[str, Any], assessment: Assessment) -> str:
    size_and_files = (
        f"**{human_size(assessment.size_bytes)}** / "
        f"{len(record.get('fileList') or [])}个文件<br>"
        f"{markdown_cell(file_summary(record))}"
    )
    recommendation = f"**{DECISION_LABELS[assessment.decision]}**<br>{assessment.reason}"
    values = [
        f"{assessment.rank}<br>`{assessment.dataset_id}`",
        dataset_display(record, assessment),
        type_display(record, assessment),
        f"{assessment.authenticity}<br>{assessment.granularity}",
        assessment.quantity,
        size_and_files,
        intro_and_annotation(record, assessment),
        access_display(record),
        popularity_display(record),
        risk_display(assessment),
        assessment.local_status,
        recommendation,
    ]
    return "| " + " | ".join(markdown_cell(value) for value in values) + " |"


def decision_summary(
    assessments: list[Assessment], decision: str
) -> tuple[int, int, int]:
    selected = [item for item in assessments if item.decision == decision]
    completed = sum(item.local_status == "已完整下载" for item in selected)
    return len(selected), sum(item.size_bytes for item in selected), completed


def exclusion_group(assessment: Assessment) -> str:
    reason = assessment.reason
    if "没有公开文件" in reason or "无公开文件" in reason:
        return "无可下载文件"
    if "重复" in reason or "转载" in reason:
        return "重复/转载"
    if contains_any(reason, ("广告", "垃圾", "恶意")):
        return "广告/垃圾内容"
    if assessment.dataset_type.startswith("非数据｜") or contains_any(
        reason, ("模型", "权重", "代码", "工具")
    ):
        return "模型/代码/工具"
    if contains_any(assessment.authenticity, ("合成", "派生", "渲染")) or contains_any(
        reason, ("合成", "派生", "原始页面")
    ):
        return "合成/派生/非原始"
    if assessment.dataset_type.startswith("非目标｜") or contains_any(
        reason, ("无关", "元数据", "纠错", "验证码绕过")
    ):
        return "非目标/元数据冲突"
    return "其他明确排除"


def render(records: list[dict[str, Any]]) -> None:
    assessments = [assess_record(record) for record in records]
    by_id = {item.dataset_id: item for item in assessments}
    counts = Counter(item.decision for item in assessments)
    type_counts = Counter(item.dataset_type for item in assessments)
    authenticity_counts = Counter(item.authenticity for item in assessments)
    total_size = sum(item.size_bytes for item in assessments)
    completed_count = sum(item.local_status == "已完整下载" for item in assessments)
    c_pairs = [
        (record, by_id[int(record["datasetId"])])
        for record in records
        if by_id[int(record["datasetId"])].decision == "C"
    ]
    c_document_count = sum(
        assessment.dataset_type.startswith("文档｜")
        for _, assessment in c_pairs
    )
    c_target_evidence_count = sum(
        contains_any(target_evidence_text(record), TARGET_EVIDENCE_TERMS)
        for record, _ in c_pairs
    )
    c_popular_count = sum(
        int(record.get("downloadCount") or 0) > 10 for record, _ in c_pairs
    )
    c_completed_count = sum(
        assessment.local_status == "已完整下载" for _, assessment in c_pairs
    )
    d_exclusion_counts = Counter(
        exclusion_group(item) for item in assessments if item.decision == "D"
    )
    local_size = sum(
        path.stat().st_size
        for path in (ROOT / "aistudio").rglob("*")
        if path.is_file()
    )
    free_space = shutil.disk_usage(ROOT).free

    decision_data = {
        "version": 1,
        "generated_at": utc_now(),
        "source_url": SOURCE_URL,
        "policy": (
            "Only decision A is eligible for automatic download. B is optional, "
            "C requires manual review, and D must be skipped."
        ),
        "summary": {
            "total": len(assessments),
            "total_size_bytes": total_size,
            "decisions": dict(sorted(counts.items())),
        },
        "auto_download_ids": [
            item.dataset_id for item in assessments if item.decision == "A"
        ],
        "datasets": [asdict(item) for item in assessments],
    }
    atomic_write_json(DECISIONS_PATH, decision_data)

    summary_rows = []
    for decision in "ABCD":
        count, size, completed = decision_summary(assessments, decision)
        summary_rows.append(
            f"| {DECISION_LABELS[decision]} | {count} | {human_size(size)} | "
            f"{completed} | {count - completed} |"
        )

    next_batch = sorted(
        [
        item
        for item in assessments
        if item.decision == "A" and item.local_status != "已完整下载"
        ],
        key=lambda item: (item.size_bytes > 10 * 2**30, item.size_bytes, item.rank),
    )
    next_rows = []
    for item in next_batch:
        record = records[item.rank - 1]
        next_rows.append(
            f"| {item.rank} | [{markdown_cell(item.name)}]"
            f"(https://aistudio.baidu.com/datasetdetail/{item.dataset_id}) | "
            f"{item.dataset_type} | {item.authenticity} | "
            f"{human_size(item.size_bytes)} | {item.reason} |"
        )

    distribution = "；".join(
        f"{kind} {count}个" for kind, count in sorted(type_counts.items())
    )
    authenticity_distribution = "；".join(
        f"{kind} {count}个" for kind, count in sorted(authenticity_counts.items())
    )
    exclusion_distribution = "；".join(
        f"{kind} {count}个" for kind, count in sorted(d_exclusion_counts.items())
    )
    header = (
        "| 原热度序/ID | 数据集与来源 | 类型/平台标签 | 真实性/粒度 | 数据量 | "
        "文件大小/构成 | 介绍/标注 | 获取与许可 | 热度 | 证据/风险 | 本地状态 | 下载建议 |"
    )
    divider = "|---:|---|---|---|---:|---|---|---|---:|---|---|---|"

    lines = [
        "# AI Studio OCR / 文档数据集下载决策分析",
        "",
        f"> 抓取与渲染日期：{date.today().isoformat()}。来源：[AI Studio OCR 数据集目录]({SOURCE_URL})。详情缓存：`{CACHE_PATH.name}`；机器决策：`{DECISIONS_PATH.name}`。",
        "",
        "## 决策口径",
        "",
        f"- 完整抓取 **{len(assessments)} 个**公开条目的详情、文件清单、平台许可和热度；详情接口标称文件合计 **{human_size(total_size)}**。当前本地已完整下载 **{completed_count} 个**。",
        f"- `aistudio` 目录当前占用 **{human_size(local_size)}**，所在磁盘可用 **{human_size(free_space)}**。报告只做决策，不会自动删除已下载内容或启动下载。",
        "- **A｜优先下载**：真实或有较强真实证据的整页/PDF、票据表单、版面、表格结构、文档VQA/KIE数据。只有 A 类进入自动下载候选。",
        "- **B｜按需下载**：真实场景文字、文本行/字符、车牌/仪表/验证码、手写和公式等OCR辅助数据；它们不是当前“原始整页文档”核心。",
        "- **C｜人工复核**：名字相关但描述、标注、粒度或来源不足，以及合成/渲染/裁剪/派生的OCR或文档图像。信息不足或非真实本身不能作为进入 D 的理由；未看样例前不下载。",
        "- **D｜不再下载**：模型权重/代码工具、普通视觉或文本任务、无公开文件、明确广告/垃圾内容、明显转载镜像。已下载的 D 类不会在本报告生成时自动删除。",
        "- `protocolName` 是上传者在平台选择的声明，不能替代原始数据来源条款。尤其是 CC0、其他、未声明以及转载条目，投入训练前仍需复核。",
        "- “文件大小”是压缩文件总量，不是解压后空间；压缩包名称无法证明内部含原始图片或标注，因此证据不足的条目会落入 C。",
        "",
        "## 汇总结论",
        "",
        "| 决策 | 数据集数 | 平台文件合计 | 已完整下载 | 未完整下载 |",
        "|---|---:|---:|---:|---:|",
        *summary_rows,
        "",
        f"- 类型分布：{distribution}。",
        f"- 真实性分布：{authenticity_distribution}。",
        f"- C 类复核池：文档型 **{c_document_count} 个**，名称/介绍/文件名有独立OCR或文档证据 **{c_target_evidence_count} 个**，平台下载量大于10的 **{c_popular_count} 个**，本地已完整下载可直接抽样的 **{c_completed_count} 个**。",
        f"- D 类明确排除原因：{exclusion_distribution}。这些条目不再因介绍短或标注缺失而进入 D。",
        "- 本目录虽然统一挂在 OCR 标签下，但标签污染严重。决策优先依赖详情描述、文件说明和文件构成，不依赖名称中的 `OCR` 字样。",
        "",
        "## 下载顺序与执行边界",
        "",
        "- `原热度序`是平台 `orderType=1` 接口返回的目录次序；原下载器按这个目录顺序遍历，并不会再按使用数、浏览数或收藏数计算一个新的热度分。平台排序只作参考，不能证明来源、许可或数据质量。",
        "- 本报告的下一批顺序不照搬热度：只选 A 类，先放置小于 10 GiB 的条目，再按压缩文件大小从小到大排列，最后才用原热度序打破同大小项的并列。这样更符合当前磁盘余量和先验样本验证需求。",
        f"- 恢复下载时应把机器清单 `{DECISIONS_PATH.name}` 中的 `auto_download_ids` 作为唯一自动下载白名单；现有下载器不会自动读取该文件，不能直接无筛选续跑。B 只按明确任务需求选取，C 必须先看样例，D 后续跳过。报告生成本身不会启动下载。",
        "- 已经完整下载但被判为 B/C/D 的目录不会自动删除，以免误删用户数据；后续释放空间时应另做清单和校验。",
        "",
        "## 分析方法与限制",
        "",
        "- 每个条目综合名称、详情介绍、平台标签、文件名/文件说明、文件大小、许可声明和热度统计分类；对已知基准、比赛数据、重复镜像、模型包、合成集及元数据冲突项另做人工纠正。",
        "- D 只收有明确排除证据的条目：无公开文件、与目标无关、模型/代码、垃圾上传、元数据冲突或确认重复。相关的合成/派生OCR或文档图进入 C，不能仅因非真实、介绍短或标注缺失而判为不再下载。",
        "- C 内优先复核平台 `downloadCount > 10` 或明确含整页文档、票据、证照、表单、试卷、古籍、版面图片的条目；下载量只是复核优先级，不改变真实性判断。",
        "- 疑似重复仅依据完整文件大小序列签名，是风险提示而非内容哈希结论；报告只列关联 ID，不自动指定主条目。最终去重应比较解压清单或文件哈希。",
        "- 本轮没有批量解压 857 个压缩包查看样例。名称与介绍无法证明真实性、但也没有排除证据的条目保守归 C；只有发现明确排除证据才归 D。文件内部是否缺失、损坏、混入模型，仍应在实际下载前抽样检查。",
        "- 数据量优先从上传者介绍提取，支持中英文常见单位；仍未说明时显示平台文件数，不能把压缩文件个数当作图片/页面数量。",
        "",
        "## 下一批优先项",
        "",
        "以下列出全部尚未完整下载的 A 类条目，先按文件小于 10 GiB、再按大小排序；`原热度序`保留平台原始顺序，便于同时参考热度。实际启动前仍应结合剩余磁盘和许可逐个确认。",
        "",
        "| 原热度序 | 数据集 | 类型 | 真实性 | 大小 | 选择理由 |",
        "|---:|---|---|---|---:|---|",
        *next_rows,
        "",
        "## 字段说明",
        "",
        "- `真实性/粒度`：区分真实采集、公开基准、合成/派生和未知，并标明整页、表格区域、场景图、文本行或专用目标。",
        "- `证据`：高/中/低由介绍长度、数据量、标注/格式/来源说明和文件描述共同计算；它衡量元数据完整度，不代表图像质量。",
        "- `本地状态`：来自 `aistudio/.aistudio-batch-state.json`，用于避免重复下载；失败和中断条目可能保留 `.part`。",
        "- `文件构成`：列出最多 7 个最大文件；完整原始字段保留在详情缓存，全部分类结果保留在机器决策 JSON。",
        "",
        "## 数据集明细",
        "",
    ]

    for decision in "ABCD":
        count, size, _ = decision_summary(assessments, decision)
        lines.extend(
            [
                f"### {DECISION_LABELS[decision]}（{count}个，{human_size(size)}）",
                "",
                header,
                divider,
            ]
        )
        for record in records:
            assessment = by_id[int(record["datasetId"])]
            if assessment.decision == decision:
                lines.append(render_row(record, assessment))
        lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Rendered {len(assessments)} datasets to {OUTPUT_PATH}; "
        f"decisions={dict(sorted(counts.items()))}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    payload = collect(args.refresh, args.workers)
    print(
        f"Cached {len(payload['details'])} details in {CACHE_PATH}",
        flush=True,
    )
    if args.collect_only:
        return
    render(build_records())


if __name__ == "__main__":
    main()
