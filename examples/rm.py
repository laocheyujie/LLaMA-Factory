import json
import logging
import os
import re

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def save_jsonl(save_path, data):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


format = "%(asctime)s.%(msecs)03d %(levelname)s [%(threadName)s] [%(filename)s:%(lineno)d] - %(message)s"
logging.basicConfig(level=logging.INFO, format=format)
logger = logging.getLogger(__name__)


class RewardRequest(BaseModel):
    model: str = "model"
    messages: list[str]

class RewardResponse(BaseModel):
    scores: float

app = FastAPI()


def extract_response_content(messages: list[str]) -> list[dict]:
    chat_history = []
    for message in messages:
        # <|im_start|>user\n你是一个擅长中文写作的 AI 助手。请根据我提供的标题和大纲，撰写一篇结构清晰、内容具体、逻辑严密的中文文章。请严格按照 Markdown 格式输出，每个大纲点作为一级段落展开。\n\n- 字数要求约为 940 字\n- 每个大纲点需用 `##` 二级标题展开为一段正文\n- 所有大纲必须覆盖，不得遗漏\n\n---\n\n【标题】\n数字化管理助力企业高效运营\n\n【大纲】\n- 数字化管理的定义与意义\n- 企业运营数字化转型路径\n- 智能工具应用与数据驱动\n\n---\n\n【输出格式】\n# 数字化管理助力企业高效运营\n## 数字化管理的定义与意义\n（请在此写出详细内容）\n## 企业运营数字化转型路径\n（请在此写出详细内容）\n## 智能工具应用与数据驱动\n（请在此写出详细内容）<|im_end|>
        query = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", message, re.DOTALL).group(1).strip()
        response = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", message, re.DOTALL).group(1).strip()
        if "<think>" in response or "</think>" in response:
            response = response.rsplit("</think>", 1)[-1].strip()
        chat_history.append({"query": query, "response": response})
    return chat_history

def get_direct_score(q_title: str, res: str) -> float:
    if res.startswith("# " + q_title):
        return 0.5
    else:
        return -0.5

def get_title_score(q: str, res: str) -> float:
    if not res:
        return -0.5
    if q:
        if q == res:
            return 0.5
        else:
            return -0.5
    else:
        print(f"q is empty, res is {res}")
        return 0.0

def get_outline_score(q: str, res: str) -> float:
    if q:
        q = set(q)
        res = set(res)
        match_count = len(q & res)
        if match_count == len(q):
            return 2.0
        # elif match_count == 0:
        #     return -2.0
        else:
            return -2.0 * (1 - (match_count / len(q)))
    else:
        print(f"q is empty, res is {res}")
        return 0.0

def get_total_count_score(q: int, res: int) -> float:
    if q == -1:
        return 1.0
    diff = abs(res - q)
    if diff <= 10:
        return 1.0
    elif diff <= 50:
        return 0.6
    elif diff <= 100:
        return 0.1
    elif diff <= 150:
        return 0
    elif diff <= 200:
        return -0.5
    elif diff <= 300:
        return -0.8
    else:
        return -1.0

def extract_info(text: str, role='user') -> dict:
    info = {
        "title": "",
        "outlines": "",
        "total_count": -1,
    }
    if role == 'user':
        match = re.search(r"###输出大纲\n# (?P<title>.*?)[\(（]?\n(?P<outlines>## .*?)(?:\n\n|###)", text, re.DOTALL)
        if match:
            info["title"] = match.group("title").strip()
            outlines = match.group("outlines").strip()
            info["outlines"] = [outline.replace("## ", "").strip() for outline in outlines.split("\n")]
        count_match = re.search(r"###输出字数要求\n.*?(?P<total_count>\d+)\S+$", text, re.DOTALL)
        if count_match:
            info["total_count"] = int(count_match.group("total_count"))
    else:
        title_match = re.search(r"^# (?P<title>.*?)$", text, re.MULTILINE)
        if title_match:
            info["title"] = title_match.group("title").strip()
        outlines = re.findall(r"^## (?P<outlines>.*?)$", text, re.MULTILINE)
        info["outlines"] = [outline.replace("## ", "").strip() for outline in outlines]
        info["total_count"] = count_text_statistics(text)
    return info


def count_text_statistics(text):
    # 统计中文字符（不包含标点）
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    chinese_char_count = len(chinese_chars)

    # 统计英文单词和数字（连续的数字视作一个词）
    english_words_and_numbers = re.findall(r'\b[a-zA-Z]+\b|\b\d+\b', text)
    english_word_number_count = len(english_words_and_numbers)

    # 统计标点符号（中英文标点符号都统计）
    punctuation_marks = re.findall(r'[\u3000-\u303F\uFF00-\uFFEF.,;:\'"!?()-]', text)
    punctuation_count = len(punctuation_marks)

    total_count = chinese_char_count + english_word_number_count + punctuation_count

    return total_count


def get_score(chat_history: list[dict]) -> float:
    scores = []
    for item in chat_history:
        q = extract_info(item["query"], role='user')
        res = extract_info(item["response"], role='assistant')
        direct_score = get_direct_score(q["title"], item["response"])
        title_score = get_title_score(q["title"], res["title"])
        outline_score = get_outline_score(q["outlines"], res["outlines"])
        total_count_score = get_total_count_score(q["total_count"], res["total_count"])
        total_score = direct_score + title_score + outline_score + total_count_score
        total_score = round(total_score, 2)
        scores.append(total_score)
    return scores


@app.post("/reward")
def reward(request: RewardRequest):
    chat_history = extract_response_content(request.messages)
    scores = get_score(chat_history)
    res = {
        "scores": scores,
    }
    for message, score in zip(chat_history, scores):
        log_data = {}
        log_data['query'] = message['query']
        log_data['response'] = message['response']
        log_data['score'] = score
        save_jsonl("./rm.jsonl", log_data)
    logger.info(res)
    return res


if __name__ == "__main__":
    # messages = [
    #     '<|im_start|>user\n请根据我提供的“输出大纲”，按照“输出格式要求”撰写一篇结构清晰、内容具体、逻辑严密的文章。\n\n###输出大纲\n# 智慧校园平台运营模式创新\n## 智慧校园功能构架\n## 运营新模式探讨\n## 案例复盘\n\n###输出格式要求\n# 文章标题\n## 大纲一\n大纲一对应的正文内容\n## 大纲二\n大纲二对应的正文内容\n## 大纲三\n大纲三对应的正文内容\n\n\n###输出字数要求\n输出的文章长度：750字左右\n<|im_end|>\n<|im_start|>assistant\n<think>\n好的，用户让我根据他们的“输出大纲”写一篇结构清晰、内容具体、逻辑严密的文章，长度大约750字左右。首先，我需要仔细分析用户提供的大纲内容，确保每个部分都涵盖到位。用户提到了“智慧校园功能构架”、“运营新模式探讨”和“案例复盘”三个主要部分，每个部分下还分大纲一、大纲二和大纲三，对应不同的正文内容。\n\n首先，我得确认每个大纲的内容是否完整。比如，智慧校园功能构架部分需要描述功能模块，运营新模式探讨要讨论现有模式的问题和创新点，案例复盘则需要实际案例的分析。用户可能希望文章既有理论又有实际例子，所以需要平衡这些方面。\n\n接下来，字数要求是750字左右，所以每个部分可能需要控制在150到200字之间，确保整体结构紧凑。标题要吸引人，可能用“智慧校园平台的创新运营模式”这样的表述。然后，开头需要引入主题，说明智慧校园的重要性，引出文章的主旨。\n\n在智慧校园功能构架部分，我需要详细说明各个功能模块，比如教学管理、学生服务、科研平台等，每个模块的功能描述要具体，比如教学管理中的智能排课系统和数据分析工具。这样能让读者清楚了解现有功能的局限性。\n\n运营新模式探讨部分，用户可能希望讨论当前存在的问题，比如传统模式的效率低、功能单一等，然后提出创新模式，比如模块化架构和数据驱动的决策。需要解释这些新模式如何解决现有问题，提升用户体验。\n\n案例复盘部分，用户可能希望引用实际成功的案例，比如某大学的应用，说明模式的实际效果。需要选择具有代表性的案例，分析其成效，并总结经验教训。\n\n最后，确保文章结构清晰，每个大纲对应一个段落，没有重复或遗漏。检查字数，确保符合要求，同时保持逻辑严密，内容具体。可能还需要调整用词，使语言更流畅，避免重复，使文章读起来更自然。\n</think>\n\n# 智慧校园平台的创新运营模式\n\n## 大纲一  \n**智慧校园功能构架**  \n智慧校园平台需构建涵盖教学、科研、学生服务和安全管理等核心功能的架构体系。具体功能包括智能排课系统（基于AI算法优化课程安排）、校园安防监控平台（实时追踪异常行为）、智能健康管理模块（通过大数据分析学生健康数据）、数字化图书馆服务（支持在线借阅和电子签名），以及校园综合信息门户。现有功能常因模块划分不合理或技术协同不足导致效率低下，例如数据孤岛问题和用户体验碎片化。\n\n## 大纲二  \n**运营新模式探讨**  \n当前智慧校园运营以"功能迭代+用户个性化服务"为核心。传统模式存在三大痛点：一是功能模块开发周期长，导致平台利用率低；二是数据孤岛问题制约决策效率，形成运营成本高、资源分散的恶性循环；三是用户需求响应滞后，难以满足多样化教育场景。创新模式应聚焦以下三个方向：  \n1. **模块化架构**：通过将核心功能拆解为可复用的微服务，实现平台的灵活扩展与快速迭代，降低开发成本。  \n2. **数据驱动决策**：建立统一的数据平台，整合多源数据，通过机器学习实现精准化运营，提升资源分配效率。  \n3. **用户定制化服务**：采用智能推荐算法和个性化数据分析，满足不同年级、专业、师生群体的需求。\n\n## 大纲三  \n**案例复盘**  \n某省重点大学引入智慧校园平台后，运营效率显著提升。通过建设"功能共融+数据共享"的创新模式，该平台的智能排课系统将课程安排时间从平均3天缩短至1.5天，学生满意度提升23%。在科研平台建设方面，通过AI辅助文献检索系统，科研人员查找到文献效率提高40%。复盘发现，该案例成功规避了传统模式的效率瓶颈，验证了创新运营模式的有效性。\n\n（全文共749字）<|im_end|>',
    #     '<|im_start|>user\n请根据我提供的“输出大纲”，按照“输出格式要求”撰写一篇结构清晰、内容具体、逻辑严密的文章。\n\n###输出大纲\n# 绿色包装材料研发与推广难题\n## 绿色包装技术瓶颈\n## 材料研发创新实例\n\n###输出格式要求\n# 文章标题\n## 大纲一\n大纲一对应的正文内容\n## 大纲二\n大纲二对应的正文内容\n## 大纲三\n大纲三对应的正文内容\n\n\n###输出字数要求\n输出的文章长度：750字左右\n<|im_end|>\n<|im_start|>assistant\n<think>\n好的，我现在需要帮用户写一篇符合他们输出大纲和格式要求的文章，大约750字。首先，我得仔细分析用户提供的大纲结构和要求。\n\n用户的大纲分为三个部分：绿色包装技术瓶颈、材料研发创新实例。输出格式要求是标题、大纲一、大纲二、大纲三，每个部分对应具体的正文内容。字数要求是750字左右。我需要确保文章结构清晰，内容具体，逻辑严密。\n\n首先，标题要准确，可能需要结合绿色包装技术瓶颈和材料创新。然后大纲一对应技术瓶颈，这里可能需要讨论现有技术的问题，比如成本高、环保性不足、易污染等。材料研发实例部分，用户可能希望有具体例子，比如新型复合材料、可降解材料等，或者结合实际应用案例。\n\n接下来，我需要检查每个大纲部分的内容是否具体，有没有足够的例子和数据支持。可能用户需要引用一些研究或案例来增强说服力。同时，注意逻辑顺序，先讲问题，再讲解决办法，最后总结趋势。\n\n还要注意字数，大约750字，所以每个部分可能需要大约250-300字左右。需要确保每个大纲对应的内容详细，不遗漏关键点。另外，用户可能希望文章不仅说明问题，还要展示进展和未来方向，所以结尾部分可以提到未来的发展趋势。\n\n现在需要确保文章符合格式要求，没有使用Markdown，而是自然的段落和标题。检查有没有重复或冗余的信息，调整语言使其更流畅。最后，通读一遍，确认符合所有要求。\n</think>\n\n# 绿色包装材料研发与推广难题\n\n## 大纲一  \n**绿色包装技术瓶颈**  \n随着全球对环境友好型包装需求的提升，传统塑料包装材料的环保性和成本优势面临显著挑战。现有绿色包装技术主要集中在材料升级和工艺优化上，但实际应用中仍存在多方面的局限性。例如，高成本的生物基材料难以大规模商业化，导致价格偏高；传统有机材料的降解速度慢，难以满足环保要求；以及包装材料与消费者使用的可持续性之间存在脱节问题。据联合国可持续发展目标（SDGs）报告，全球每年约有5000万吨塑料垃圾进入海洋，而当前可回收材料利用率不足3%，说明包装行业亟需突破技术瓶颈。\n\n## 大纲二  \n**材料研发创新实例**  \n为突破上述瓶颈，多项创新技术正在推动绿色包装材料的发展。例如，生物基复合材料的研发显著提高了材料的强度与耐久性。2022年，美国科学家通过将玉米淀粉与玉米蛋白结合，开发出一种新型生物基包装膜，其回收率达90%，且成本比传统聚乙烯材料降低40%。此外，纳米材料的应用也在提升材料性能。荷兰某科研机构研发的纳米级改性聚丙烯（PP）膜，其表面具有抗紫外线和阻燃特性，可延长包装寿命至5年以上。这些创新实例表明，材料研发需要结合生物可降解性与工程性能的双重优化。\n\n## 大纲三  \n**推广路径与未来展望**  \n当前，绿色包装材料的推广面临多重挑战，包括政策支持不足、市场接受度较低以及技术转化效率有限。然而，随着全球对循环经济理念的推广，未来发展趋势具有积极前景。例如，欧盟提出"循环经济包装计划"，旨在通过回收再利用提高材料利用率，而日本则推动生物基材料在包装行业的产业化应用。此外，智能包装技术的发展也为绿色材料的推广提供了新机遇。结合物联网和人工智能，新型智能包装系统能动态调节包装材料的性质，延长使用寿命并降低资源消耗。综上所述，绿色包装材料的研发与推广需要在技术创新、政策引导和市场应用等多个层面协同推进，以实现可持续发展目标的长期实现。<|im_end|>'
    # ]
    # request = RewardRequest(messages=messages)

    uvicorn.run(app, host="0.0.0.0", port=8018)
