#!/usr/bin/env python3
"""Collect and review ModelScope datasets selected by the image-to-text filter."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import requests


SOURCE_URL = (
    "https://www.modelscope.cn/datasets?"
    "Tags=image-to-text&dataType=image&page=1"
)
ANYOCR_COLLECTION_URL = (
    "https://www.modelscope.cn/collections/AnyOCR-4987429313b046"
)
ANYOCR_COLLECTION_DATASET_COUNT = 17
LIST_API = "https://www.modelscope.cn/api/v1/dolphin/datasets"
DETAIL_API = "https://www.modelscope.cn/api/v1/datasets/{owner}/{name}"
DATA_SOURCES_DIR = Path(__file__).resolve().parents[1] / "data_sources"
CACHE_PATH = DATA_SOURCES_DIR / ".modelscope_ocr_cache.json"
DEFAULT_OUTPUT_PATH = DATA_SOURCES_DIR / "modelscope数据集OCR.md"
DOWNLOAD_STATE_PATH = Path(r"E:\data\doc\modelscope\modelscope-download-state.json")
PAGE_SIZE = 30


# AnyOCR contains 17 datasets. Three already occur in the image-to-text result;
# these 14 records complete the union, including the previously requested
# Layout-Instruction-Data dataset.
SUPPLEMENTAL_DATASETS = [
    {
        "Owner": "iic",
        "Name": "Layout-Instruction-Data",
        "ChineseName": "文档理解版式指令微调数据集",
        "License": "Apache License 2.0",
        "Description": "LayoutLLM (CVPR 2024) 指令微调数据集",
        "UserDefineTags": "Document Understanding,Instruction Tuning,Document AI,OCR",
        "StorageSize": 28_318_058_439,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": (
            "LayoutLLM data includes document images, OCR results, 116k layout-aware "
            "pre-training descriptions, 300k supervised fine-tuning instructions with "
            "and without LayoutCoT, and CORD/FUNSD/SROIE evaluation data."
        ),
        "WebUrl": "https://www.modelscope.cn/datasets/iic/Layout-Instruction-Data",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "VirtualLUO",
        "Name": "Chronicles-OCR",
        "ChineseName": "Chronicles-OCR 古文字跨时代评测集",
        "License": "Apache License 2.0",
        "Description": "覆盖汉字七种书体演变的跨时代视觉感知评测集。",
        "UserDefineTags": "OCR,Chinese scripts,benchmark",
        "StorageSize": 1_320_573_834,
        "ApprovalMode": 1,
        "ProtectedMode": 2,
        "ReadmeContent": "2,800 balanced images spanning the Seven Chinese Scripts.",
        "WebUrl": "https://www.modelscope.cn/datasets/VirtualLUO/Chronicles-OCR",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "nv-community",
        "Name": "OCR-Synthetic-Multilingual-v1",
        "ChineseName": "NVIDIA 多语言合成 OCR 数据集",
        "License": "cc-by-4.0",
        "Description": "面向多语言文字检测与识别的大规模 SynthDoG 合成数据。",
        "UserDefineTags": "synthetic-data,hdf5,text-detection,ocr,text-recognition",
        "StorageSize": 5_448_080_361_912,
        "ApprovalMode": 1,
        "ProtectedMode": 2,
        "ReadmeContent": "12,258,146 samples in six languages with word, line, and paragraph annotations.",
        "WebUrl": "https://www.modelscope.cn/datasets/nv-community/OCR-Synthetic-Multilingual-v1",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "kevin726",
        "Name": "hy_202504_ocr_data",
        "ChineseName": "hy_202504 OCR 数据",
        "License": "Apache License 2.0",
        "Description": "AnyOCR 合集收录的大型 OCR 仓库，数据卡未提供有效说明。",
        "UserDefineTags": "OCR",
        "StorageSize": 299_432_725_476,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": "Default dataset card; inspect files and samples after download.",
        "WebUrl": "https://www.modelscope.cn/datasets/kevin726/hy_202504_ocr_data",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "DaoCloud",
        "Name": "daocloud-datasets",
        "ChineseName": "DaoCloud 文档微调数据集",
        "License": "Apache License 2.0",
        "Description": "使用 DeepSeek 模型蒸馏得到的 DaoCloud 文档微调数据。",
        "UserDefineTags": "kubernetes,daocloud,document",
        "StorageSize": 83_653_157,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": "DaoCloud documentation fine-tuning data distilled with DeepSeek.",
        "WebUrl": "https://www.modelscope.cn/datasets/DaoCloud/daocloud-datasets",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "iic",
        "Name": "MP-DocStruct1M",
        "ChineseName": "MP-DocStruct1M 多页文档理解数据集",
        "License": "Apache License 2.0",
        "Description": "DocOwl2 多页文档理解预训练数据，包含解析和页码查找任务。",
        "UserDefineTags": "Document Understanding,OCR,multi-page",
        "StorageSize": 108_930_998_187,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": "One million multi-page document parsing and page retrieval samples.",
        "WebUrl": "https://www.modelscope.cn/datasets/iic/MP-DocStruct1M",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "iic",
        "Name": "DocStruct4M",
        "ChineseName": "DocStruct4M",
        "License": "Apache License 2.0",
        "Description": "包含文档、网页、表格、图表和自然图的统一文档结构学习数据。",
        "UserDefineTags": "文字识别和定位,4M,文档图片解析",
        "StorageSize": 338_883_088_410,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": "Three million structured parsing and one million text localization/recognition samples.",
        "WebUrl": "https://www.modelscope.cn/datasets/iic/DocStruct4M",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "iic",
        "Name": "D4LA",
        "ChineseName": "D4LA 版面分析数据集",
        "License": "Apache License 2.0",
        "Description": "覆盖多类文档与细粒度版面元素的文档版面分析数据集。",
        "UserDefineTags": "Document Layout Analysis,Document AI,OCR",
        "StorageSize": 1_387_984_290,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": "11,092 document pages across 12 document categories and 27 layout categories.",
        "WebUrl": "https://www.modelscope.cn/datasets/iic/D4LA",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "racineai",
        "Name": "ocr-pdf-degraded",
        "ChineseName": "OCR-PDF-Degraded",
        "License": "apache-2.0",
        "Description": "合成退化文档图像与对应 OCR 真值文本。",
        "UserDefineTags": "OCR,PDF,degraded documents",
        "StorageSize": 2_688_759_806,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": "Synthetically degraded document images paired with OCR ground truth.",
        "WebUrl": "https://www.modelscope.cn/datasets/racineai/ocr-pdf-degraded",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "prithivMLmods",
        "Name": "Corvus-OCR-Caption-Mini-Mix",
        "ChineseName": "Corvus OCR Caption Mini Mix",
        "License": "apache-2.0",
        "Description": "混合 OCR、科学文档、数学内容与长描述的图文训练集。",
        "UserDefineTags": "image,document,caption,OCR,LaTeX",
        "StorageSize": 850_300_911,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": "A compact English/Chinese image-caption mix with OCR-heavy samples.",
        "WebUrl": "https://www.modelscope.cn/datasets/prithivMLmods/Corvus-OCR-Caption-Mini-Mix",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "allenai",
        "Name": "olmOCR-mix-0225",
        "ChineseName": "olmOCR Mix 0225",
        "License": "Apache License 2.0",
        "Description": "PDF 页面及按自然阅读顺序生成的纯文本 OCR 训练数据。",
        "UserDefineTags": "OCR,PDF,document parsing",
        "StorageSize": 56_055_609_869,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": "266,135 pages from 105,504 documents, OCRed with GPT-4o.",
        "WebUrl": "https://www.modelscope.cn/datasets/allenai/olmOCR-mix-0225",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "ChatDOC",
        "Name": "OCRFlux-bench-single",
        "ChineseName": "OCRFlux 单页文档解析评测集",
        "License": "Apache License 2.0",
        "Description": "中英文 PDF 页面与人工复核 Markdown 真值的单页解析评测集。",
        "UserDefineTags": "OCR,PDF,benchmark,Markdown",
        "StorageSize": 661_076_030,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": "2,000 PDF pages: 1,000 Chinese and 1,000 English.",
        "WebUrl": "https://www.modelscope.cn/datasets/ChatDOC/OCRFlux-bench-single",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "ChatDOC",
        "Name": "OCRFlux-pubtabnet-single",
        "ChineseName": "OCRFlux PubTabNet 表格解析评测集",
        "License": "Apache License 2.0",
        "Description": "PubTabNet 表格图像及转换后的 HTML 真值。",
        "UserDefineTags": "OCR,table,benchmark,HTML",
        "StorageSize": 202_639_953,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": "9,064 table images: 4,623 simple and 4,441 complex tables.",
        "WebUrl": "https://www.modelscope.cn/datasets/ChatDOC/OCRFlux-pubtabnet-single",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    },
    {
        "Owner": "AI-ModelScope",
        "Name": "LaTeX_OCR",
        "ChineseName": "LaTeX OCR 数据集",
        "License": "Apache License 2.0",
        "Description": "印刷体、合成手写和真实手写数学公式图像及 LaTeX 标注。",
        "UserDefineTags": "OCR,LaTeX,formula,handwriting",
        "StorageSize": 1_110_020_241,
        "ApprovalMode": None,
        "ProtectedMode": 2,
        "ReadmeContent": "Five formula OCR subsets including about 100k printed and 100k synthetic handwritten samples.",
        "WebUrl": "https://www.modelscope.cn/datasets/AI-ModelScope/LaTeX_OCR",
        "Supplemental": True,
        "SupplementalSource": "AnyOCR",
    }
]

# This repository is already present in the requested download root and will be
# hash-verified and reused by modelscope_batch_downloader.py.
LOCAL_REUSE_KEYS = {"iic/Layout-Instruction-Data"}


TYPE_OVERRIDES = {
    "iic/Layout-Instruction-Data": "文档｜理解/指令微调",
    "VirtualLUO/Chronicles-OCR": "OCR｜手写文字",
    "nv-community/OCR-Synthetic-Multilingual-v1": "OCR｜场景文字",
    "kevin726/hy_202504_ocr_data": "OCR｜信息不足",
    "DaoCloud/daocloud-datasets": "文档｜理解/指令微调",
    "iic/MP-DocStruct1M": "文档｜预训练语料",
    "iic/DocStruct4M": "文档｜预训练语料",
    "iic/D4LA": "文档｜版面分析",
    "racineai/ocr-pdf-degraded": "OCR｜场景文字",
    "prithivMLmods/Corvus-OCR-Caption-Mini-Mix": "多模态｜图文/VQA",
    "allenai/olmOCR-mix-0225": "文档｜预训练语料",
    "ChatDOC/OCRFlux-bench-single": "评测｜OCR/文档",
    "ChatDOC/OCRFlux-pubtabnet-single": "OCR｜表格/公式",
    "AI-ModelScope/LaTeX_OCR": "OCR｜表格/公式",
    "WIRD9090/ocr_plate": "OCR｜票据/专用",
    "Qwen/CC-OCR": "评测｜OCR/文档",
    "iic/ICDAR13_HCTR_Dataset": "OCR｜手写文字",
    "zacbi2023/coco2017_caption": "多模态｜图文/VQA",
    "modelscope/ocr_fudanvi_zh": "OCR｜场景文字",
    "liekkas/table_recognition": "OCR｜表格/公式",
    "Wente47/M2E": "OCR｜表格/公式",
    "HiDolphin/MaritimeOCRBench": "评测｜OCR/文档",
    "iic/MTWI": "OCR｜场景文字",
    "Genius-Society/svhn": "OCR｜票据/专用",
    "liekkas/text_det_test_dataset": "OCR｜文本检测",
    "meituan-longcat/UNO-Bench": "多模态｜图文/VQA",
    "iic/SIBR": "文档｜解析/KIE",
    "Kpillow/SceneVTG-Erase": "视觉｜文字生成/擦除",
    "AI-ModelScope/idl-wds": "文档｜预训练语料",
    "swift/llava-med-zh-instruct-60k": "多模态｜图文/VQA",
    "iic/WebText_Dataset": "OCR｜场景文字",
    "xmatrix/OCR_Synthetic_LaTeX": "OCR｜表格/公式",
    "megemini/OCR-KIE": "文档｜解析/KIE",
    "iic/TUL": "OCR｜场景文字",
    "mitchell/ocr_drug": "OCR｜票据/专用",
    "OpenGVLab/OmniCorpus-YT": "多模态｜图文/VQA",
    "AI-ModelScope/chinese_text_recognition": "OCR｜场景文字",
    "jikeyang/ODIR-5K": "视觉｜其他",
    "Leofyfan/GSM8K-V": "多模态｜图文/VQA",
    "Amorter/captcha_chinese_click_1": "OCR｜票据/专用",
    "AI-ModelScope/pdfa-eng-wds": "文档｜预训练语料",
    "market.aliyun/IMG_JP_OCR_Invoices_CN": "OCR｜票据/专用",
    "market.aliyun/IMG_GA_OCR_CN": "OCR｜场景文字",
    "market.aliyun/IMG_KOR_OCR_CN": "OCR｜场景文字",
    "tf4444/my_dataset": "OCR｜场景文字",
    "shhzxt/clock1": "视觉｜其他",
    "wuwuwuwuwuwuwuwu/ocr_demo": "其他｜信息不足",
    "market.aliyun/IMG_OCR_ARU002_CN": "OCR｜场景文字",
    "market.aliyun/IMG_OCR_Vietnam_CN": "OCR｜场景文字",
}


QUANTITY_OVERRIDES = {
    "iic/Layout-Instruction-Data": "116,000条预训练描述 + 300,000条SFT指令",
    "VirtualLUO/Chronicles-OCR": "2,800张 / 汉字七种书体",
    "nv-community/OCR-Synthetic-Multilingual-v1": "12,258,146条 / 6种语言",
    "kevin726/hy_202504_ocr_data": "README未说明",
    "DaoCloud/daocloud-datasets": "README未说明",
    "iic/MP-DocStruct1M": "约1,000,000条多页文档样本",
    "iic/DocStruct4M": "约4,000,000条（解析3M+文字定位/识别1M）",
    "iic/D4LA": "11,092页 / 12类文档 / 27类版面元素",
    "racineai/ocr-pdf-degraded": "README未说明",
    "prithivMLmods/Corvus-OCR-Caption-Mini-Mix": "README未说明（仅train划分）",
    "allenai/olmOCR-mix-0225": "105,504文档 / 266,135页",
    "ChatDOC/OCRFlux-bench-single": "2,000页（中英文各1,000页）",
    "ChatDOC/OCRFlux-pubtabnet-single": "9,064张表格图",
    "AI-ModelScope/LaTeX_OCR": "5个子集；印刷体约10万+合成手写约10万+真实手写",
    "WIRD9090/ocr_plate": "约410,000张（训练37万+验证4万）",
    "Qwen/CC-OCR": "7,058张 / 39个子集",
    "iic/ICDAR13_HCTR_Dataset": "3,432张",
    "zacbi2023/coco2017_caption": "123,287张图；每图约5条描述",
    "modelscope/ocr_fudanvi_zh": "README未说明",
    "liekkas/table_recognition": "18张",
    "Wente47/M2E": "99,956张",
    "HiDolphin/MaritimeOCRBench": "1,888条 / 5类任务",
    "iic/MTWI": "20,000张（训练/测试各半）",
    "Genius-Society/svhn": "超过600,000张数字图",
    "liekkas/text_det_test_dataset": "23张",
    "meituan-longcat/UNO-Bench": "1,250条全模态 + 2,480条单模态",
    "iic/SIBR": "1,000张（训练600+测试400）",
    "Kpillow/SceneVTG-Erase": "155,000张 / 约192万文本行",
    "AI-ModelScope/idl-wds": "3,144,726文档 / 19,174,595页",
    "swift/llava-med-zh-instruct-60k": "约60,000条",
    "iic/WebText_Dataset": "10,000张文本切片",
    "xmatrix/OCR_Synthetic_LaTeX": "2,000条合成样本",
    "megemini/OCR-KIE": "9,088条 / 7个来源集",
    "iic/TUL": "4,800张",
    "mitchell/ocr_drug": "333张",
    "OpenGVLab/OmniCorpus-YT": "1,000万篇图文交错文档",
    "AI-ModelScope/chinese_text_recognition": "README未说明",
    "jikeyang/ODIR-5K": "5,000名患者 / 约10,000张眼底图",
    "Leofyfan/GSM8K-V": "1,319题 / 5,343张图",
    "Amorter/captcha_chinese_click_1": "README未说明",
    "AI-ModelScope/pdfa-eng-wds": "2,159,432文档 / 约1,800万页",
    "market.aliyun/IMG_JP_OCR_Invoices_CN": "992张",
    "market.aliyun/IMG_GA_OCR_CN": "11,248张",
    "market.aliyun/IMG_KOR_OCR_CN": "3,724张",
    "tf4444/my_dataset": "269张",
    "shhzxt/clock1": "500张（训练400+测试100）",
    "wuwuwuwuwuwuwuwu/ocr_demo": "README未说明",
    "market.aliyun/IMG_OCR_ARU002_CN": "15,054张",
    "market.aliyun/IMG_OCR_Vietnam_CN": "16,557张（无标注）",
    "DatatangBeijing/100People-HandwritingOCRDataofJapaneseandKorean": "100人 / 22,163张",
    "DatatangBeijing/28972Images_Driver_Face_Detection_Face_96_Landmarks_Annotation_Data": "100人 / 28,972张",
    "DatatangBeijing/41605People_Multiple_Styles_Video_Data": "41,605人；每人4-50段视频",
    "DatatangBeijing/1044million_English_Test_Questions_Text_Parsing_And_Processing_Data": "约1,044万道",
    "DatatangBeijing/155People_Malay_Speech_Data_by_Mobile_Phone_Guiding": "155人（时长未说明）",
    "DatatangBeijing/262People-5162ImagesHandwritingOCRDataofTraditionalChineseCharactersTaiwanChina": "262人 / 5,162张",
    "DatatangBeijing/1000People-SpanishHandwritingOCRData": "1,000人 / 14,000张",
    "DatatangBeijing/1000People-FrenchHandwritingOCRData": "1,000人 / 14,000张",
    "DatatangBeijing/Mandarin_Chinese_Multi_Stream_Speech_Dataset_294_Speakers_203_Hours": "294人 / 203小时",
    "DatatangBeijing/INTERSPEECH_2025_MLC_SLM_Challenge_Dataset": "15套来源集；竞赛子集量未说明",
    "DatatangBeijing/4_People_Chinese_High_expressivity_Narration_Average_Tone_Speech_Synthesis_Corpus": "4人（时长未说明）",
    "DatatangBeijing/20011ImageCaptionDataOfOCRInNaturalScenes": "20,011张（正文称20,000张）",
    "DatatangBeijing/1000People-ItalianHandwritingOCRData": "1,000人 / 14,000张",
    "DatatangBeijing/1000People-GermanHandwritingOCRData": "1,000人 / 14,000张",
    "DatatangBeijing/351_People_German_Speech_Data_by_Mobile_Phone_Guiding": "351人（时长未说明）",
}


INTRO_OVERRIDES = {
    "iic/Layout-Instruction-Data": "LayoutLLM文档理解语料，含文档图、OCR结果、版式预训练描述、带/不带LayoutCoT的SFT指令及CORD/FUNSD/SROIE评测数据。",
    "VirtualLUO/Chronicles-OCR": "面向甲骨文至现代书体演变的跨时代中文视觉感知评测集，2,800张图按汉字七种书体严格平衡。",
    "nv-community/OCR-Synthetic-Multilingual-v1": "NVIDIA基于扩展SynthDoG生成的六语种OCR训练集，HDF5内含图像、词/行/段框、四边形和阅读顺序图。",
    "kevin726/hy_202504_ocr_data": "AnyOCR收录的278.87 GiB大型OCR仓库，但数据卡仍是默认模板，任务、标注和来源均需下载后抽检。",
    "DaoCloud/daocloud-datasets": "使用DeepSeek蒸馏得到的DaoCloud文档微调数据；更接近文档问答/知识微调，因AnyOCR收录而列为必下。",
    "iic/MP-DocStruct1M": "DocOwl2多页文档理解预训练集，覆盖多页文字解析和根据文字查找页码两类任务。",
    "iic/DocStruct4M": "DocOwl1.5统一文档结构学习数据，覆盖文档、网页、表格、图表和自然图，含解析及多粒度文字定位/识别。",
    "iic/D4LA": "细粒度文档版面分析数据，覆盖12类文档与27类版面元素，提供图像、检测JSON及VGT网格特征。",
    "racineai/ocr-pdf-degraded": "由干净PDF页面合成透视、模糊、亮度、对比度和JPEG退化，并配对OCR真值与退化参数。",
    "prithivMLmods/Corvus-OCR-Caption-Mini-Mix": "英中图文混合集，兼有自然图长描述、OCR密集科学/数学/文档样本及LaTeX内容。",
    "allenai/olmOCR-mix-0225": "网页PDF和Internet Archive图书页面，经GPT-4o按自然阅读顺序生成纯文本，可训练或评估文档OCR管线。",
    "ChatDOC/OCRFlux-bench-single": "人工多轮复核的中英文PDF页面与Markdown真值，用于单页OCR和版面解析评测。",
    "ChatDOC/OCRFlux-pubtabnet-single": "由PubTabNet转换得到的表格图与HTML真值，覆盖简单表格和含跨行/跨列单元格的复杂表格。",
    "AI-ModelScope/LaTeX_OCR": "来自公开公式资源和自建数据的五个公式OCR子集，覆盖印刷体、合成手写、真实手写及对应印刷版本。",
    "WIRD9090/ocr_plate": "车牌识别训练/验证集；页面声称约41万张，但仓库实际只有JSON和README。",
    "Qwen/CC-OCR": "覆盖场景文字、多语言、文档解析和KIE的综合OCR评测集，附VLMEvalKit用TSV。",
    "iic/ICDAR13_HCTR_Dataset": "ICDAR 2013中文手写文本识别公开评测集；仓库主要是CSV索引。",
    "zacbi2023/coco2017_caption": "COCO 2017图像描述标注，不是OCR；图像需从COCO官网另行下载。",
    "modelscope/ocr_fudanvi_zh": "FudanVI中文场景文字识别数据镜像，含train/val/test说明；仓库仅少量索引。",
    "liekkas/table_recognition": "18张有线/无线表格图及HTML真值，用于表格还原算法快速评测。",
    "Wente47/M2E": "真实试卷和练习册中的多行数学公式图，提供LaTeX标注及train/val/test。",
    "HiDolphin/MaritimeOCRBench": "海事及通用文档的IE、VQA、解析和文字定位综合评测，JSONL指令格式。",
    "iic/MTWI": "中英网络图像场景文字检测与识别，四边形框加文本转写，版式复杂。",
    "Genius-Society/svhn": "Google街景门牌数字识别/检测数据，适合数字串任务，不覆盖通用文本。",
    "liekkas/text_det_test_dataset": "23张自然场景图和LabelMe标注，用于文本检测指标的轻量回归测试。",
    "meituan-longcat/UNO-Bench": "图像、视频、音频和文本联合问答评测，不是OCR数据集。",
    "iic/SIBR": "中英自然场景视觉信息抽取，含实体、框、实体内/间链接，兼容FUNSD/XFUND风格。",
    "Kpillow/SceneVTG-Erase": "原图、文字擦除图和文本行标注，用于视觉文字生成/擦除而非传统OCR。",
    "AI-ModelScope/idl-wds": "工业文档PDF/TIFF与Textract OCR标注的WebDataset语料，适合大规模文档预训练。",
    "swift/llava-med-zh-instruct-60k": "由LLaVA-Med翻译得到的中文医学视觉指令数据，不是OCR专用集。",
    "iic/WebText_Dataset": "从MTWI抽取的中英文本行切片，用于通用/场景文字识别测试。",
    "xmatrix/OCR_Synthetic_LaTeX": "LLM生成Markdown/LaTeX后渲染成图的合成公式OCR对，覆盖度和公式正确性需复核。",
    "megemini/OCR-KIE": "统一整理发票、收据、表单和营养标签等7个KIE数据源，适合结构化信息抽取。",
    "iic/TUL": "长度2-25字符均匀分布的英文场景文字评测集，专测长度外推鲁棒性。",
    "mitchell/ocr_drug": "PPOCRLabel手工标注的药品外包装文字小集，用于药名识别实验。",
    "OpenGVLab/OmniCorpus-YT": "从YouTube收集的大规模图文交错文档语料，不是OCR标注集。",
    "AI-ModelScope/chinese_text_recognition": "FudanVI中文文本识别数据镜像；README只给来源，需自行核对划分和许可。",
    "jikeyang/ODIR-5K": "眼底疾病分类图像压缩包，与OCR/文档识别无关。",
    "Leofyfan/GSM8K-V": "把GSM8K题目渲染为多场景图的视觉数学推理评测，不是公式OCR。",
    "Amorter/captcha_chinese_click_1": "VOC XML格式的中文点选验证码框标注，README未交代样本数。",
    "AI-ModelScope/pdfa-eng-wds": "英文PDF及词/行/图像框OCR元数据的WebDataset语料，面向大规模文档预训练。",
    "market.aliyun/IMG_JP_OCR_Invoices_CN": "日语收据、报价单和订单图像，平台只提供部分样例。",
    "market.aliyun/IMG_GA_OCR_CN": "港澳广告、牌匾、菜单、地图和商铺等场景图，含标注与未标注部分。",
    "market.aliyun/IMG_KOR_OCR_CN": "韩语广告、牌匾、菜单、地图和商铺等场景图，平台只提供部分样例。",
    "tf4444/my_dataset": "269张印刷体图片的个人OCR小集，采用LMDB组织，适合流程冒烟测试。",
    "shhzxt/clock1": "模拟指针钟表图片及hour/minute/second标签，是结构化视觉读数而非文字OCR。",
    "wuwuwuwuwuwuwuwu/ocr_demo": "默认模板且仓库只有数百字节，未提供可分析的数据内容。",
    "market.aliyun/IMG_OCR_ARU002_CN": "阿拉伯语广告、地图、包装、标语和店铺牌图像，平台只提供部分样例。",
    "market.aliyun/IMG_OCR_Vietnam_CN": "越南语菜单、列表、地图、包装、店招和手写图像，无标注且仅提供样例。",
    "DatatangBeijing/41605People_Multiple_Styles_Video_Data": "人物多风格视频，标称41,605人、每人4-50段；README后文又称2.5万人，规模口径需向供应方确认。",
    "DatatangBeijing/5Hours_Shanghai_Dialect_Speech_Synthesis_Corpus_Female": "上海方言单人TTS录音；标题写女声，README正文却写男声，购买前需确认说话人性别。",
    "DatatangBeijing/INTERSPEECH_2025_MLC_SLM_Challenge_Dataset": "2025 MLC-SLM多语种对话语音竞赛集，来自15套对话语音，面向重叠、打断和长上下文ASR。",
    "DatatangBeijing/5Hours_Changsha_Dialect_Speech_Synthesis_Corpus_Female": "长沙方言单人TTS录音；标题写女声，README正文却写男声，购买前需确认说话人性别。",
    "DatatangBeijing/600_Hours_Norwegian_Real_world_Casual_Conversation_and_Monologue_speech_dataset": "挪威语口语化ASR语料；README又称录音人为罗马尼亚人，语种/人员元数据需向供应方确认。",
}


RECOMMENDATION_OVERRIDES = {
    "iic/Layout-Instruction-Data": "下载：本地已有26.37 GiB仓库；哈希校验后直接复用",
    "VirtualLUO/Chronicles-OCR": "必下（AnyOCR新增）：古文字跨时代OCR评测；需申请访问",
    "nv-community/OCR-Synthetic-Multilingual-v1": "必下（AnyOCR新增），本轮暂缓：六语种合成OCR；4.96 TiB且需申请访问",
    "kevin726/hy_202504_ocr_data": "必下（AnyOCR新增）：先抽检任务、来源和标注；278.87 GiB独立批次",
    "DaoCloud/daocloud-datasets": "必下（AnyOCR新增）：文档微调数据；核对是否包含图像/OCR字段",
    "iic/MP-DocStruct1M": "必下（AnyOCR新增），用户确认已有：E/F/G指定目录未检出，待提供路径核验；本轮不重复下载",
    "iic/DocStruct4M": "必下（AnyOCR新增），用户确认已有：E/F/G指定目录未检出，待提供路径核验；本轮不重复下载",
    "iic/D4LA": "必下（AnyOCR新增）：版面分析训练与评测",
    "racineai/ocr-pdf-degraded": "必下（AnyOCR新增）：退化文档鲁棒OCR训练与评测",
    "prithivMLmods/Corvus-OCR-Caption-Mini-Mix": "必下（AnyOCR新增）：OCR密集图文预训练；注意其同时含普通长描述样本",
    "allenai/olmOCR-mix-0225": "必下（AnyOCR新增），其他电脑已有：本机不下载；F盘仅有不完整残留",
    "ChatDOC/OCRFlux-bench-single": "必下（AnyOCR新增）：中英文单页文档解析评测",
    "ChatDOC/OCRFlux-pubtabnet-single": "必下（AnyOCR新增）：表格图到HTML解析评测",
    "AI-ModelScope/LaTeX_OCR": "必下（AnyOCR新增）：印刷体与手写公式OCR",
    "WIRD9090/ocr_plate": "下载：仓库仅索引，原图需另找上游",
    "Qwen/CC-OCR": "下载：综合OCR/文档评测",
    "iic/ICDAR13_HCTR_Dataset": "下载：作为ICDAR备份；先验证索引原图",
    "modelscope/ocr_fudanvi_zh": "下载：先取索引，再核对外部原图",
    "liekkas/table_recognition": "下载：轻量表格回归测试",
    "Wente47/M2E": "下载：多行公式识别",
    "HiDolphin/MaritimeOCRBench": "下载：需申请访问",
    "iic/MTWI": "下载：先取索引并核对外部原图占用",
    "Genius-Society/svhn": "下载：数字/门牌文字识别",
    "liekkas/text_det_test_dataset": "下载：轻量文本检测评测",
    "iic/SIBR": "下载：KIE训练与评测",
    "Kpillow/SceneVTG-Erase": "本地已有：G盘29/29个远端文件、322.79 GiB核验完整，无需重复下载",
    "AI-ModelScope/idl-wds": "下载：PDF/TIFF原文档；超大批次并审查版权",
    "iic/WebText_Dataset": "下载：识别评测；核对外部原图",
    "xmatrix/OCR_Synthetic_LaTeX": "下载：公式OCR；先抽检合成质量",
    "megemini/OCR-KIE": "下载：含文档图/KIE；需申请访问",
    "iic/TUL": "下载：文字长度鲁棒性评测",
    "mitchell/ocr_drug": "下载：药品包装文字；核对外部原图",
    "AI-ModelScope/chinese_text_recognition": "下载：先核对许可及与FudanVI重复",
    "Amorter/captcha_chinese_click_1": "下载：验证码文字与框标注",
    "AI-ModelScope/pdfa-eng-wds": "下载：PDF原文档与OCR框；超大批次",
    "tf4444/my_dataset": "下载：印刷体OCR小集",
    "wuwuwuwuwuwuwuwu/ocr_demo": "下载核验：仓库近空壳，无有效训练数据",
}


METADATA_ONLY = {
    "WIRD9090/ocr_plate",
    "iic/ICDAR13_HCTR_Dataset",
    "modelscope/ocr_fudanvi_zh",
    "iic/MTWI",
    "liekkas/text_det_test_dataset",
    "iic/WebText_Dataset",
    "mitchell/ocr_drug",
    "wuwuwuwuwuwuwuwu/ocr_demo",
}


ANYOCR_NEW_KEYS = {
    "VirtualLUO/Chronicles-OCR",
    "nv-community/OCR-Synthetic-Multilingual-v1",
    "kevin726/hy_202504_ocr_data",
    "DaoCloud/daocloud-datasets",
    "iic/MP-DocStruct1M",
    "iic/DocStruct4M",
    "iic/D4LA",
    "racineai/ocr-pdf-degraded",
    "prithivMLmods/Corvus-OCR-Caption-Mini-Mix",
    "allenai/olmOCR-mix-0225",
    "ChatDOC/OCRFlux-bench-single",
    "ChatDOC/OCRFlux-pubtabnet-single",
    "AI-ModelScope/LaTeX_OCR",
}


ANYOCR_UNLOCATED_USER_OWNED_KEYS = {
    "iic/MP-DocStruct1M",
    "iic/DocStruct4M",
}


ANYOCR_OTHER_COMPUTER_KEYS = {
    "allenai/olmOCR-mix-0225",
}


ANYOCR_SKIP_KEYS = ANYOCR_UNLOCATED_USER_OWNED_KEYS | ANYOCR_OTHER_COMPUTER_KEYS


VERIFIED_LOCAL_GIANT_KEYS = {
    "Kpillow/SceneVTG-Erase",
    "kevin726/hy_202504_ocr_data",
}


ANYOCR_DEFERRED_KEYS = {
    "nv-community/OCR-Synthetic-Multilingual-v1",
}


EXTRA_DOWNLOAD_KEYS = {
    "Kpillow/SceneVTG-Erase",
    "DatatangBeijing/20011ImageCaptionDataOfOCRInNaturalScenes",
    "wuwuwuwuwuwuwuwu/ocr_demo",
    *ANYOCR_NEW_KEYS,
}


GIANT_DOWNLOAD_KEYS = {
    "Kpillow/SceneVTG-Erase",
    "AI-ModelScope/idl-wds",
    "AI-ModelScope/pdfa-eng-wds",
    "nv-community/OCR-Synthetic-Multilingual-v1",
    "kevin726/hy_202504_ocr_data",
    "iic/MP-DocStruct1M",
    "iic/DocStruct4M",
    "allenai/olmOCR-mix-0225",
}


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    attempts: int = 5,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if payload.get("Code") != 200:
                raise RuntimeError(
                    f"API returned Code={payload.get('Code')}: {payload.get('Message')}"
                )
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def fetch_list(session: requests.Session) -> tuple[list[dict[str, Any]], int]:
    base_params = {
        "PageSize": PAGE_SIZE,
        "PageNumber": 1,
        "Tags": "image:image-to-text",
    }
    first = request_json(session, LIST_API, params=base_params)
    total = int(first["TotalCount"])
    records = list(first.get("Data") or [])

    for page in range(2, math.ceil(total / PAGE_SIZE) + 1):
        params = dict(base_params, PageNumber=page)
        payload = request_json(session, LIST_API, params=params)
        records.extend(payload.get("Data") or [])

    return records, total


def fetch_detail(record: dict[str, Any]) -> dict[str, Any]:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 ModelScope dataset catalog research",
            "X-Modelscope-Visit-From": "pc",
        }
    )
    url = DETAIL_API.format(owner=record["Owner"], name=record["Name"])
    payload = request_json(session, url, params={"Revision": "master"})
    return payload.get("Data") or {}


def collect() -> dict[str, Any]:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 ModelScope dataset catalog research",
        }
    )
    records, reported_total = fetch_list(session)
    if len(records) != reported_total:
        raise RuntimeError(
            f"List endpoint reported {reported_total}, fetched {len(records)}"
        )

    details: list[dict[str, Any] | None] = [None] * len(records)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_index = {
            executor.submit(fetch_detail, record): index
            for index, record in enumerate(records)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            details[index] = future.result()
            print(
                f"[{sum(item is not None for item in details):3d}/{len(records)}] "
                f"{records[index]['Owner']}/{records[index]['Name']}",
                flush=True,
            )

    datasets = []
    for list_record, detail in zip(records, details, strict=True):
        merged = dict(list_record)
        merged.update(detail or {})
        merged["WebUrl"] = (
            f"https://www.modelscope.cn/datasets/"
            f"{merged['Owner']}/{merged['Name']}"
        )
        datasets.append(merged)

    return {
        "source_url": SOURCE_URL,
        "list_api": LIST_API,
        "filter": {"Tags": "image:image-to-text"},
        "reported_total": reported_total,
        "datasets": datasets,
    }


def normalized_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("Name"),
        record.get("ChineseName"),
        record.get("Description"),
        record.get("UserDefineTags"),
        record.get("ReadmeContent"),
    ]
    text = "\n".join(str(part) for part in parts if part)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[[^]]*]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"[#*`_|>-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def dataset_key(record: dict[str, Any]) -> str:
    return f"{record['Owner']}/{record['Name']}"


def classify(record: dict[str, Any]) -> str:
    key = dataset_key(record)
    if key in TYPE_OVERRIDES:
        return TYPE_OVERRIDES[key]

    name = record["Name"].lower()
    title = str(record.get("ChineseName") or "")
    text = f"{name} {title}"
    if "speech_synthesis" in name or "合成库" in title:
        return "语音｜TTS"
    if "noise" in name or "噪音" in title:
        return "音频｜噪声"
    if "interspeech" in name or "speech" in name or "语音" in title:
        return "语音｜ASR"
    if "video" in name or "视频" in title:
        return "视频｜生成/理解"
    if "caption" in name or "图文描述" in title:
        return "多模态｜图文/VQA"
    if "test_questions" in name or "fine_tuning" in name:
        return "文本｜NLP/LLM"
    if "face" in name or "portrait" in name or "人像" in title:
        return "视觉｜其他"
    if "primaryschoolmathematics" in name:
        return "文档｜图像/试卷"
    if "formula" in name:
        return "OCR｜表格/公式"
    if "forms" in name:
        return "OCR｜表格/公式"
    if "handwriting" in name or "手写" in title:
        return "OCR｜手写文字"
    if "invoice" in name or "发票" in title:
        return "OCR｜票据/专用"
    if "ocr" in name or "ocr" in text.lower():
        return "OCR｜场景文字"
    return "其他｜信息不足"


def quantity(record: dict[str, Any]) -> str:
    key = dataset_key(record)
    if key in QUANTITY_OVERRIDES:
        return QUANTITY_OVERRIDES[key]

    candidates = [record.get("ChineseName"), record.get("Description"), record["Name"]]
    pattern = re.compile(
        r"((?:\d{1,3}(?:[,，]\d{3})+|\d+(?:\.\d+)?)\s*(?:万|亿)?\s*"
        r"(?:小时|张|组|人|道|条|段|套))",
        flags=re.IGNORECASE,
    )
    for candidate in candidates:
        if not candidate:
            continue
        match = pattern.search(str(candidate))
        if match:
            return match.group(1).replace("，", ",").replace(" ", "")
    return "README未说明"


def concise_description(record: dict[str, Any], limit: int = 105) -> str:
    key = dataset_key(record)
    if key in INTRO_OVERRIDES:
        return INTRO_OVERRIDES[key]

    text = re.sub(r"\s+", " ", str(record.get("Description") or "")).strip()
    if not text:
        text = normalized_text(record)
    for prefix in (record["Name"], str(record.get("ChineseName") or "")):
        if prefix and text.startswith(prefix):
            text = text[len(prefix) :].lstrip(" ：:-—。，,；;")
    if len(text) <= limit:
        return text or "页面未提供有效介绍。"
    cut = max(text.rfind("。", 0, limit), text.rfind("；", 0, limit))
    if cut >= 35:
        return text[: cut + 1]
    return text[:limit].rstrip("，,；; ") + "…"


def normalize_license(value: Any) -> str:
    if not value or value == [] or value == "[]":
        return "未声明"
    return str(value)


def access_note(record: dict[str, Any]) -> str:
    owner = record["Owner"]
    license_name = normalize_license(record.get("License"))
    if owner == "DatatangBeijing":
        return f"商业；全量需采购（仓库字段：{license_name}）"
    if owner == "market.aliyun":
        return "仅部分样例；定制条款/完整集需提交需求"
    suffix = "；需申请访问" if record.get("ApprovalMode") == 1 else "；可直接访问"
    return license_name + suffix


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


def repository_size(record: dict[str, Any]) -> str:
    key = dataset_key(record)
    size = human_size(record.get("StorageSize"))
    if record["Owner"] in {"DatatangBeijing", "market.aliyun"}:
        return size + "（样例/页面仓库）"
    if key in METADATA_ONLY:
        return size + "（元数据/索引）"
    if key == "zacbi2023/coco2017_caption":
        return size + "（仅标注，图像另下）"
    return size


def should_download(record: dict[str, Any], dataset_type: str) -> bool:
    key = dataset_key(record)
    return dataset_type.startswith(("OCR｜", "文档｜", "评测｜OCR")) or key in EXTRA_DOWNLOAD_KEYS


def recommendation(record: dict[str, Any], dataset_type: str) -> str:
    key = dataset_key(record)
    selected = should_download(record, dataset_type)
    if not selected:
        return "不下载：不含目标OCR/文档数据"
    if key in RECOMMENDATION_OVERRIDES:
        return RECOMMENDATION_OVERRIDES[key]

    if record["Owner"] == "DatatangBeijing":
        return "下载：先取样例；完整集需采购"
    if record["Owner"] == "market.aliyun":
        return "下载：先取样例；完整集需提交需求"
    if record.get("ApprovalMode") == 1:
        return "下载：需申请访问"
    if normalize_license(record.get("License")) in {"未声明", "other"}:
        return "下载：先审查许可"
    if key in METADATA_ONLY:
        return "下载：先取索引并核对外部原图"
    return "下载：可直接获取"


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def download_execution_lines() -> list[str]:
    if not DOWNLOAD_STATE_PATH.exists():
        return []
    try:
        state = json.loads(DOWNLOAD_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []

    datasets = state.get("datasets") or {}
    counts = Counter(item.get("status", "unknown") for item in datasets.values())
    labels = {
        "completed": "公开仓库完成",
        "sample_downloaded": "商业页面样例完成",
        "existing_verified": "已有仓库校验完成",
        "pending_access": "待登录/申请",
        "failed": "失败",
    }
    summary = "；".join(
        f"{labels.get(status, status)} {count} 个"
        for status, count in sorted(counts.items())
    )
    newly_written = sum(
        int(item.get("local_bytes") or 0)
        for key, item in datasets.items()
        if item.get("status") in {"completed", "sample_downloaded"}
        and key not in LOCAL_REUSE_KEYS
    )
    pending = [
        key
        for key, item in datasets.items()
        if item.get("status") not in {"completed", "sample_downloaded", "existing_verified"}
    ]
    pending_text = "、".join(f"`{key}`" for key in pending) or "无"
    log_path = state.get("last_log") or "未记录"
    return [
        "## 当前下载执行结果",
        "",
        f"> 状态更新时间：{state.get('updated_at', '未知')}。",
        "",
        f"- 本批共 {state.get('target_count', len(datasets))} 个目标：{summary}。",
        f"- 本次新写入约 **{human_size(newly_written)}**；`iic/Layout-Instruction-Data` 复用并 SHA-256 校验了已有的 26.37 GiB。E 盘剩余 **{human_size(int(state.get('free_bytes') or 0))}**。",
        f"- 尚未取得：{pending_text}。`Wente47/M2E` 的匿名仓库接口返回 HTTP 401，需配置 ModelScope Token/完成登录后用 `--retry-access` 续传。",
        f"- 下载日志：`{log_path}`；逐项状态：`{DOWNLOAD_STATE_PATH.with_name('modelscope-download-status.md')}`。",
        "- 数据堂和云市场的 29 项仅下载了页面仓库样例，不代表已购买或取得商业全量。",
        "",
    ]


def render_markdown(payload: dict[str, Any], output_path: Path) -> None:
    rows = []
    type_counts: Counter[str] = Counter()
    all_size = 0
    selected_count = 0
    selected_repo_size = 0
    selected_direct_count = 0
    selected_direct_size = 0
    selected_direct_without_giants = 0
    selected_local_reuse_size = 0
    selected_giant_size = 0
    selected_commercial_count = 0
    selected_commercial_page_size = 0

    for index, record in enumerate(payload["datasets"], 1):
        key = dataset_key(record)
        dataset_type = classify(record)
        type_counts[dataset_type] += 1
        storage_size = int(record.get("StorageSize") or 0)
        all_size += storage_size
        selected = should_download(record, dataset_type)
        is_commercial_listing = record["Owner"] in {"DatatangBeijing", "market.aliyun"}
        if selected:
            selected_count += 1
            selected_repo_size += storage_size
            if is_commercial_listing:
                selected_commercial_count += 1
                selected_commercial_page_size += storage_size
            else:
                selected_direct_count += 1
                selected_direct_size += storage_size
                if key in GIANT_DOWNLOAD_KEYS:
                    selected_giant_size += storage_size
                else:
                    selected_direct_without_giants += storage_size
                    if key in LOCAL_REUSE_KEYS:
                        selected_local_reuse_size += storage_size

        display = key
        if record.get("ChineseName"):
            display += f"<br>{record['ChineseName']}"
        link = f"[{markdown_cell(display)}]({record['WebUrl']})"
        rows.append(
            "| {index} | {link} | {kind} | {quantity} | {size} | {intro} | {access} | {advice} |".format(
                index=index,
                link=link,
                kind=markdown_cell(dataset_type),
                quantity=markdown_cell(quantity(record)),
                size=markdown_cell(repository_size(record)),
                intro=markdown_cell(concise_description(record)),
                access=markdown_cell(access_note(record)),
                advice=markdown_cell(recommendation(record, dataset_type)),
            )
        )

    distribution = "；".join(
        f"{kind} {count}个" for kind, count in sorted(type_counts.items())
    )
    today = date.today().isoformat()
    analyzed_total = len(payload["datasets"])
    supplemental_count = analyzed_total - int(payload["reported_total"])
    new_direct_bytes = selected_direct_without_giants - selected_local_reuse_size
    new_transfer_bytes = new_direct_bytes + selected_commercial_page_size
    anyocr_new_records = [
        record
        for record in payload["datasets"]
        if dataset_key(record) in ANYOCR_NEW_KEYS
    ]
    anyocr_new_size = sum(
        int(record.get("StorageSize") or 0) for record in anyocr_new_records
    )
    anyocr_largest_key = "nv-community/OCR-Synthetic-Multilingual-v1"
    anyocr_largest_size = next(
        int(record.get("StorageSize") or 0)
        for record in anyocr_new_records
        if dataset_key(record) == anyocr_largest_key
    )
    anyocr_current_batch_records = [
        record
        for record in anyocr_new_records
        if dataset_key(record)
        not in ANYOCR_SKIP_KEYS | ANYOCR_DEFERRED_KEYS
    ]
    anyocr_current_batch_size = sum(
        int(record.get("StorageSize") or 0)
        for record in anyocr_current_batch_records
    )
    anyocr_unlocated_names = "、".join(
        f"`{key}`" for key in sorted(ANYOCR_UNLOCATED_USER_OWNED_KEYS)
    )
    anyocr_other_computer_names = "、".join(
        f"`{key}`" for key in sorted(ANYOCR_OTHER_COMPUTER_KEYS)
    )
    anyocr_deferred_names = "、".join(
        f"`{key}`" for key in sorted(ANYOCR_DEFERRED_KEYS)
    )
    giant_pending_keys = (
        GIANT_DOWNLOAD_KEYS - ANYOCR_SKIP_KEYS - VERIFIED_LOCAL_GIANT_KEYS
    )
    giant_pending_size = sum(
        int(record.get("StorageSize") or 0)
        for record in payload["datasets"]
        if dataset_key(record) in giant_pending_keys
    )
    giant_names = "、".join(f"`{key}`" for key in sorted(giant_pending_keys))
    giant_verified_names = "、".join(
        f"`{key}`" for key in sorted(VERIFIED_LOCAL_GIANT_KEYS)
    )
    lines = [
        "# ModelScope 数据集 OCR / 文档识别筛选分析",
        "",
        f"> 抓取日期：{today}。来源：[ModelScope 筛选页]({payload['source_url']})；[AnyOCR 合集]({ANYOCR_COLLECTION_URL})。",
        "",
        "## 口径与结论",
        "",
        f"- 当前筛选接口返回 **{payload['reported_total']} 个**数据集；AnyOCR 合集含 **{ANYOCR_COLLECTION_DATASET_COUNT} 个**数据集，其中原报告已有 4 个，本次新增 **{len(anyocr_new_records)} 个必下项**。与筛选结果去重后共分析 **{analyzed_total} 个**（补充 {supplemental_count} 个）。",
        f"- 类型统计：{distribution}。标签 `image-to-text` 存在明显误标，语音、噪声、视频、纯文本和普通视觉数据均被混入。",
        "- 下载原则：凡是 OCR、文档解析/识别、图像文字信息提取、含文档图，或提供 PDF/TIFF 等原始文档的条目均下载；AnyOCR 合集本次新增的 13 项按用户要求全部列为必下。商业、受控、索引型和超大规模只影响获取方式与批次，不影响入选。",
        f"- 按此原则应下载 **{selected_count} 个**，不下载 **{analyzed_total - selected_count} 个**。入选页面仓库合计 **{human_size(selected_repo_size)}**；其中可直接访问或申请访问的 {selected_direct_count} 个仓库合计 **{human_size(selected_direct_size)}**。",
        f"- 排除 {len(GIANT_DOWNLOAD_KEYS)} 个超大仓库后，其余非商业入选仓库合计 **{human_size(selected_direct_without_giants)}**。其中 `iic/Layout-Instruction-Data` 的 **{human_size(selected_local_reuse_size)}** 已在目标盘，校验后复用；加上商业页面样例，普通批次预计新增传输 **{human_size(new_transfer_bytes)}**，按1.2倍预留 **{human_size(math.ceil(new_transfer_bytes * 1.2))}**。",
        f"- {len(GIANT_DOWNLOAD_KEYS)} 个超大仓库本身合计 **{human_size(selected_giant_size)}**，按1.2倍预留 **{human_size(math.ceil(selected_giant_size * 1.2))}**。全部 {analyzed_total} 个已分析页面仓库合计 **{human_size(all_size)}**。",
        f"- AnyOCR 新增 {len(anyocr_new_records)} 个仓库合计 **{human_size(anyocr_new_size)}**；其中 `{anyocr_largest_key}` 单项 **{human_size(anyocr_largest_size)}**，其余 12 项仍有 **{human_size(anyocr_new_size - anyocr_largest_size)}**。G 盘总容量 447.12 GiB、检查时可用 113.93 GiB，即使清空也无法容纳这 13 项。",
        f"- 本轮执行：{anyocr_deferred_names} 暂不下载；{anyocr_other_computer_names} 在其他电脑已有，本机不下载；{anyocr_unlocated_names} 由用户确认已持有，但在 E/F/G 指定目录未检出，待提供路径核验且本轮不重复下载；其余 {len(anyocr_current_batch_records)} 项下载或续传至 `F:\\modelscope`，平台标称合计 **{human_size(anyocr_current_batch_size)}**。",
        "- E/F/G 三个指定目录共命中 61 个入选仓库：60 个已按远端清单核验或修复完成；`allenai/olmOCR-mix-0225` 的 F 盘副本不完整，因完整副本在其他电脑而停止本机续传。`Kpillow/SceneVTG-Erase` 在 G 盘核验为 29/29 个文件、322.79 GiB；`kevin726/hy_202504_ocr_data` 在 F 盘核验为 28/28 个文件、278.87 GiB。",
        f"- 入选的 {selected_commercial_count} 个数据堂/云市场条目当前页面仓库仅合计 **{human_size(selected_commercial_page_size)}**，但这通常只是展示文件或样例，完整商业数据的真实大小未公开。",
        "- “文件大小”来自详情接口的仓库存储量，不等于解压后占用。标为“元数据/索引”的条目可能在加载时继续下载外部图片；数据堂和云市场条目通常只存展示文件或样例，**完整商业数据的真实大小未公开，不能据此做全量硬盘预算**。",
        "- 商业条目页面即使显示 Apache-2.0，也同时在 README 声明“商用数据/版权归数据堂”，因此表中按更保守的商业获取口径处理。",
        "",
        "## 建议下载顺序",
        "",
        f"1. 第一批：校验复用 `iic/Layout-Instruction-Data`，再下载其余 {selected_direct_count - len(GIANT_DOWNLOAD_KEYS) - len(LOCAL_REUSE_KEYS)} 个非商业普通规模入选仓库，预计新增 **{human_size(new_direct_bytes)}**。其中索引型条目加载后可能继续拉取原图；受控仓库需先完成申请。",
        f"2. 第二批：获取 {selected_commercial_count} 个数据堂/云市场入选条目的样例并逐一询价/提交需求；页面样例仓库合计 **{human_size(selected_commercial_page_size)}**，不能代表全量。",
        f"3. 第三批：对尚未持有的超大仓库逐项安排 {giant_names}，仓库合计 **{human_size(giant_pending_size)}**；{giant_verified_names} 已在本地核验完整，无需重复下载；{anyocr_other_computer_names} 在其他电脑已有，本机不下载；{anyocr_unlocated_names} 由用户确认已持有，本轮不重复下载，但尚待提供实际路径完成核验。其余项目仍须分别安排磁盘、带宽、解压空间、访问申请和许可审查。",
        f"4. 其余 {analyzed_total - selected_count} 个语音、噪声、视频、普通图像/图文、视觉问答和纯文本条目不下载。",
        "",
        *download_execution_lines(),
        "## 数据集明细",
        "",
        "| # | 数据集 | 类型 | 数据量 | 文件大小 | 介绍 / 标注 | 获取与许可 | 是否下载 / 获取建议 |",
        "|---:|---|---|---:|---:|---|---|---|",
        *rows,
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def show_records(payload: dict[str, Any], start: int, count: int) -> None:
    datasets = payload["datasets"]
    stop = min(len(datasets), start - 1 + count)
    for index in range(start - 1, stop):
        record = datasets[index]
        tags = record.get("UserDefineTags") or ""
        excerpt = normalized_text(record)[:700]
        print(f"\n[{index + 1}] {record['Owner']}/{record['Name']}")
        print(f"中文名: {record.get('ChineseName') or ''}")
        print(f"自定义标签: {tags}")
        print(f"摘要: {excerpt}")


def load_or_collect(refresh: bool) -> dict[str, Any]:
    if refresh or not CACHE_PATH.exists():
        payload = collect()
        CACHE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    result = dict(payload)
    result["datasets"] = list(payload["datasets"])
    existing = {dataset_key(record) for record in result["datasets"]}
    result["datasets"].extend(
        dict(record)
        for record in SUPPLEMENTAL_DATASETS
        if dataset_key(record) not in existing
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    args = parser.parse_args()
    payload = load_or_collect(args.refresh)
    if args.render:
        render_markdown(payload, args.output)
        print(f"Rendered {len(payload['datasets'])} rows to {args.output}")
        return
    print(
        f"Source total: {payload['reported_total']}; "
        f"cached datasets: {len(payload['datasets'])}"
    )
    show_records(payload, args.start, args.count)


if __name__ == "__main__":
    main()
