"""
意图分类器
基于规则的五分类：narrator / luxun / ambiguous / reject_time / reject_irrelevant
支持作为独立模块或 RAG 管线中的一个环节调用

用法：
    from scripts.intent_classifier import classify
    result = classify("鲁迅的原名是什么？")
    print(result)  # {"intent": "luxun", "confidence": 0.92, ...}
"""

import re
from typing import Optional

# ============================================================
# 词库定义
# ============================================================

# --- narrator 讲解模式 ---
NARRATOR_STRONG_KEYWORDS = [
    "纪念馆", "展区", "展品", "展厅", "展览", "文物",
    "参观", "游览", "开放时间", "预约", "门票", "地址",
    "在哪里", "怎么去", "怎么走", "钟楼", "白云楼",
    "陈列", "馆内", "馆藏", "导览", "讲解服务",
    "适合.*参观", "值得.*参观", "推荐.*参观",
]

NARRATOR_WEAK_PATTERNS = [
    r"(展示|介绍).*(什么|哪些|内容)",
    r"(怎么|如何).*(参观|游览|预约|去|到)",
    r"(适合|推荐).*(参观|游览|去|游玩)",
    r"(有没有|是否有).*(展|收藏|陈列|展示)",
    r"(什么|哪些).*(展|收藏|陈列)",
    r"(第一次|首次).*(参观|游览|去)",
    r"(带孩子|学生|亲子).*(参观|游览|合适)",
    r"(参观|游览).*(注意|须知|事项)",
    r"(为什么|为何).*纪念(馆|鲁迅)",
    r"(什么|何时|哪年).*建(立|馆)",
]

# --- luxun 数字人模式 ---
LUXUN_STRONG_KEYWORDS = [
    "鲁迅先生", "您觉得", "您认为", "您怎么看",
    "请用鲁迅", "用鲁迅的口吻", "用你的口吻",
    "弃医从文", "代表作", "写作风格",
    "您的作品", "您的文章", "您的小说",
    "您为什么", "您如何看待",
]

LUXUN_WEAK_PATTERNS = [
    # 基础事实类
    r"(鲁迅|他).*(原名|出生|生于|哪里|哪年|身份|被称为)",
    r"鲁迅.*的.*(原名|名字|生日|出生|身份)",
    # 生平经历类
    r"(鲁迅|他).*(一生|经历|阶段|时期|生平|留学|期间)",
    # 原因/动机类
    r"(鲁迅|他).*(为什么|为何|如何看待|认为|觉得)",
    r"(鲁迅|他).*为什么.*(要|会|选择|关注|批判)",
    # 作品相关
    r"(什么|哪些|哪部).*(作品|文章|小说|书|杂文|散文|代表作)",
    r"《.+》.*(讲了|讲述|介绍|内容|为什么|意义|主要|特点)",
    r"(鲁迅|他).*(作品|文章|小说).*(特点|特色|风格|思想)",
    r"(鲁迅|他).*(使用|采用|经常|为什么).*(讽刺|手法|笔法|写法|比喻)",
    # 思想观念
    r"(鲁迅|他).*(对|关于|如何看待).*(青年|社会|传统|文学|读书|国民|旧社会|文化)",
    r"(鲁迅|他).*(思想|精神|理念|主张|批判|态度)",
    # 虚拟假设（向鲁迅提问价值观，非时间越界）
    r"(如果|假如).*(面对|遇到|有).*(鲁迅|你).*会.*(怎么|如何|怎样|什么|建议|说|做|给出)",
    # 人称/敬语 → 在向鲁迅本人说话
    r"(请用|用).*(鲁迅|你).*(口吻|风格|语气|语言|谈谈)",
    r"(您|你).*(喜欢|讨厌|想|觉得|看待|认为|对).*",
    # 检核/事实确认
    r"(鲁迅|他).*(是否|是不是|有没有|是否曾经).*(获得|得过|写过|被|受|当选|担任|任教|是一名|是一位|是一个|从事|毕业于)",
    # 鲁迅是否是XX类型的问题
    r"鲁迅.*(是否是|是不是一名|是不是一位).*",
    # 后世影响
    r"鲁迅.*(为什么|为何).*(直到|至今|仍然|今天|依然).*(影响|重要|有|被)",
    # 作品解读
    r"鲁迅.*的.*(文学|创作|思想|作品).*(体现|反映|表达).*",
]

# --- reject_time 时间越界 ---
MODERN_TECH = [
    # 电子产品/通讯
    "手机", "电脑", "计算机", "网络", "互联网", "因特网",
    "电视", "电视机", "收音机",
    # 交通工具（1936年后普及的）
    "高铁", "地铁",
    # 社交媒体/平台
    "微信", "微博", "抖音", "B站", "bilibili", "QQ",
    "小红书", "快手", "知乎", "贴吧", "社交媒体",
    # AI/科技
    "人工智能", "AI", "ChatGPT", "机器人", "大模型",
    "机器学习", "深度学习", "神经网络", "算法推荐",
    # 军事/航天
    "原子弹", "核弹", "核武器", "卫星", "火箭", "登月", "航天",
    # 现代娱乐
    "电影院", "视频", "直播", "综艺", "电视剧",
    "电竞", "游戏", "动漫", "漫威",
    # 现代职场/社会概念（1936年后才出现的概念/词汇）
    "996", "内卷", "躺平",
    # 其他现代概念（1936年后才有的全球性活动/概念）
    "奥斯卡", "奥运会", "世界杯", "联合国",
]

# 当代人物（1949年前后出生/成名，鲁迅在1936年殁后不可能知晓）
MODERN_FIGURES = [
    # 当代华语/世界文学人物（常被拿来与鲁迅比较）
    "村上春树", "莫言", "余华", "贾平凹", "金庸", "古龙", "三毛",
    "王朔", "刘慈欣", "韩寒", "郭敬明", "王小波", "史铁生",
    "海子", "顾城", "北岛", "迟子建", "苏童", "阿来",
    # 当代科技/商业名人
    "马云", "马化腾", "李彦宏", "任正非", "雷军", "乔布斯", "马斯克",
]

FUTURE_TIME_MARKERS = [
    "新中国成立", "解放后", "建国后", "改革开放",
    "文革", "文化大革命", "大跃进",
    "二战", "第二次世界大战", "抗日胜利",
    "1949年", "1950年", "1960年", "1970年",
    "1980年", "1990年", "2000年", "2010年", "2020年",
    "现代年轻人", "当今社会", "现在的生活", "当下年轻人",
]

TIME_CLASH_PATTERNS = [
    # 鲁迅 + 现代事物
    r"(鲁迅|您|你).*(用过|见过|参加|玩过|看过|知道).*(手机|电脑|网络|电视|电影|汽车|飞机|人工智能|机器人|抖音|微信|互联网|直播|综艺|游戏|动漫|社交媒体|漫威)",
    r"(鲁迅|您|你).*(有没有|是否有|有过|是否).*(手机|电脑|网络|电视|飞机|人工智能|机器人|微信|抖音|社交媒体)",
    # 假设鲁迅活在今天
    r"(如果|假如|假设).*(鲁迅|你).*(活在|生活在|来到|穿越到).*(今天|现代|当代|现在|当下|202)",
    r"请用鲁迅的.*(评价|看待|谈谈|说说).*(现代|当今|现在|当下|如今|年轻人|996|内卷|躺平)",
    # 鲁迅 + 现代职场概念
    r"鲁迅.*(996|内卷|躺平)",
    # 纪念馆 + 不可能的东西
    r"纪念(馆|鲁迅).*(手机|人工智能|机器人|夜间|半夜|12点|二十四小时|100个|扩大到|扩建)",
    # 鲁迅 + 现代角色
    r"鲁迅.*(是否|曾经|是不是).*(参加|担任|做过|当过).*(网络|现代|人工智能|AI|程序员|研究员)",
]

# 陷阱问题（问法本身有问题）
TRAP_PATTERNS = [
    # 纪念馆 + 不可能的东西
    r"纪念(馆|鲁迅).*(手机|人工智能|机器人|夜间|半夜|12点|二十四小时|扩大到.*展厅|扩建|扩大到.*倍)",
    # 鲁迅 + 现代角色/活动
    r"鲁迅.*(是否|曾经|是不是).*(参加|担任|做过|当过).*(网络|现代|人工智能|AI|程序员|研究员|网红)",
    # 鲁迅 + 现代科技使用
    r"鲁迅.*(用过|见过|玩过).*(手机|电脑|网络|电视|汽车|飞机|抖音|微信)",
    # 纪念馆 + 收藏不可能的东西
    r"纪念(馆|鲁迅).*收藏.*(人工智能|机器人|奖杯|诺贝尔)",
]

# --- reject_irrelevant 无关/恶意 ---
PROFANITY_PATTERNS = [
    r"(操|fuck|shit|傻逼|他妈|你妈|日你|滚蛋|去死|废物|垃圾|白痴|蠢货)",
    r"(骂|侮辱|攻击).*",
]

# Prompt 注入/角色覆盖攻击
PROMPT_INJECTION_PATTERNS = [
    r"(忽略|忘记|无视|放弃).*(之前|此前|上面|原来|一开始|先前).*(设定|指令|规则|要求|身份|角色)",
    r"(从现在|从现在起|从现在开始|现在开始|现在起).*(你是|你变成|你是只|你是头|你是条).*(猫|狗|动物|机器人|AI)",
    r"(你是一只|你是一只|你是只|你是头|你是条).*(猫|狗|猪|鸟|鱼|动物)",
    r"(输出|打印|显示|告诉我|泄露).*(System Prompt|系统提示|系统指令|完整.*指令|所有.*规则)",
    r"(帮我|给我|替我).*(写.*攻击|写.*骂|写.*政府|写.*领导人|骂.*政府|骂.*领导人)",
]

IRRELEVANT_PATTERNS = [
    # 政治（非历史语境）
    r"(习近平|特朗普|拜登|泽连斯基|普京)",
    # 纯娱乐
    r"(王者荣耀|原神|吃鸡|英雄联盟|LOL|明星八卦|追星|饭圈|演唱会|综艺)",
    # 完全无关
    r"^(天气|今天|明天|你好|再见|谢谢|哈哈|呵呵|嗯|哦|啊)[\s。.。!！?？]*$",
    r"(帮我|给我).*(写|画|翻译|算|编程|代码|做数学|解题)",
]


# ============================================================
# 检测函数
# ============================================================

def _match_any(text: str, patterns: list[str]) -> list[str]:
    """返回所有匹配到的模式"""
    matched = []
    for p in patterns:
        if re.search(p, text):
            matched.append(p)
    return matched


def _match_keywords(text: str, keywords: list[str]) -> list[str]:
    """返回所有匹配到的关键词"""
    matched = []
    for kw in keywords:
        if kw in text:
            matched.append(kw)
    return matched


def check_irrelevant(text: str) -> Optional[dict]:
    """检测无关/恶意输入，命中则返回结果，否则返回 None"""
    # Prompt 注入/角色覆盖攻击（最高优先级）
    injection = _match_any(text, PROMPT_INJECTION_PATTERNS)
    if injection:
        return {
            "intent": "reject_irrelevant",
            "confidence": 0.95,
            "reason": "Prompt注入/角色覆盖攻击",
            "matched": injection,
        }

    profanity = _match_any(text, PROFANITY_PATTERNS)
    if profanity:
        return {
            "intent": "reject_irrelevant",
            "confidence": 0.95,
            "reason": "恶意输入",
            "matched": profanity,
        }

    irrelevant = _match_any(text, IRRELEVANT_PATTERNS)
    if irrelevant:
        return {
            "intent": "reject_irrelevant",
            "confidence": 0.85,
            "reason": "无关输入",
            "matched": irrelevant,
        }

    return None


def check_time_clash(text: str) -> Optional[dict]:
    """检测时间越界，命中则返回结果，否则返回 None"""
    matched = []

    # 陷阱问题 → 最高置信
    traps = _match_any(text, TRAP_PATTERNS)
    if traps:
        return {
            "intent": "reject_time",
            "confidence": 0.95,
            "reason": "陷阱/越界问题",
            "matched": traps,
        }

    # 未来时间锚点
    time_markers = _match_keywords(text, FUTURE_TIME_MARKERS)
    matched.extend(time_markers)

    # 现代科技词
    tech = _match_keywords(text, MODERN_TECH)
    matched.extend(tech)

    # 当代人物词
    figures = _match_keywords(text, MODERN_FIGURES)
    matched.extend(figures)

    # 跨越句式
    clash_patterns = _match_any(text, TIME_CLASH_PATTERNS)
    matched.extend(clash_patterns)

    if matched:
        # 置信度：现代科技词+句式 > 当代人物/纯句式 > 纯单个科技词
        if tech and clash_patterns:
            confidence = 0.92
        elif figures:
            # 当代人物名无歧义，是强时间越界信号
            confidence = 0.88
        elif clash_patterns:
            confidence = 0.88
        elif len(tech) >= 2:
            confidence = 0.85
        elif len(tech) == 1:
            # 单个科技词可能是比喻/引用，降置信
            confidence = 0.72
        else:
            confidence = 0.78
        return {
            "intent": "reject_time",
            "confidence": confidence,
            "reason": f"涉及1936年后事物/概念",
            "matched": matched,
        }

    return None


def score_narrator(text: str) -> float:
    """计算讲解模式得分"""
    score = 0.0

    strong = _match_any(text, NARRATOR_STRONG_KEYWORDS)
    score += len(strong) * 2.0

    weak = _match_any(text, NARRATOR_WEAK_PATTERNS)
    score += len(weak) * 1.0

    # bonus: 包含"广州"/"纪念馆"/"参观"任二 → +1
    venue_signals = ["广州", "纪念馆", "参观", "钟楼"]
    if sum(1 for s in venue_signals if s in text) >= 2:
        score += 1.0

    return score


def score_luxun(text: str) -> float:
    """计算数字人模式得分"""
    score = 0.0

    strong = _match_any(text, LUXUN_STRONG_KEYWORDS)
    score += len(strong) * 2.0

    weak = _match_any(text, LUXUN_WEAK_PATTERNS)
    score += len(weak) * 1.0

    # 人称信号: "您" → 很可能是向鲁迅提问
    if "您" in text:
        score += 1.5
    # 问号 + 鲁迅 → 可能是在问关于鲁迅的问题
    if "鲁迅" in text and ("？" in text or "?" in text):
        score += 0.5
    # 鲁迅著名作品名（即使没出现"鲁迅"二字）
    LU_XUN_FAMOUS_WORKS = [
        "狂人日记", "阿Q正传", "呐喊", "彷徨", "朝花夕拾", "野草",
        "孔乙己", "药", "故乡", "社戏", "祝福", "伤逝",
        "而已集", "华盖集", "三闲集", "二心集", "准风月谈",
        "故事新编", "中国小说史略", "灯下漫笔", "记念刘和珍君",
        "藤野先生", "从百草园到三味书屋", "范爱农",
    ]
    if any(w in text for w in LU_XUN_FAMOUS_WORKS):
        score += 1.0

    return score


def classify_normal(text: str) -> dict:
    """正常意图分类：narrator / luxun / ambiguous"""
    n_score = score_narrator(text)
    l_score = score_luxun(text)

    if n_score > l_score and n_score >= 1.5:
        return {
            "intent": "narrator",
            "confidence": min(n_score / 6.0, 0.95),
            "reason": "讲解模式得分领先",
            "scores": {"narrator": n_score, "luxun": l_score},
        }
    elif l_score > n_score and l_score >= 1.5:
        return {
            "intent": "luxun",
            "confidence": min(l_score / 6.0, 0.95),
            "reason": "数字人模式得分领先",
            "scores": {"narrator": n_score, "luxun": l_score},
        }
    else:
        return {
            "intent": "ambiguous",
            "confidence": 0.5,
            "reason": "无法区分或双低分",
            "scores": {"narrator": n_score, "luxun": l_score},
        }


# ============================================================
# 主入口
# ============================================================

def classify(text: str) -> dict:
    """
    对用户输入进行意图分类

    Args:
        text: 用户原始输入文本

    Returns:
        {
            "intent": str,       # narrator | luxun | ambiguous | reject_time | reject_irrelevant
            "confidence": float, # 0.0 ~ 1.0
            "reason": str,       # 分类理由
            "matched": list,     # 命中的关键词/模式（越界时有值）
            "scores": dict,      # narrator vs luxun 得分（正常分类时有值）
            "time_aware": bool,  # 是否需要时间边界提醒
        }
    """
    text = text.strip()

    # 空输入 → ambiguous
    if not text:
        return {
            "intent": "ambiguous",
            "confidence": 0.3,
            "reason": "空输入",
        }

    # Step 1: 无关/恶意
    result = check_irrelevant(text)
    if result:
        result["time_aware"] = False
        return result

    # Step 2: 时间越界
    result = check_time_clash(text)
    if result:
        result["time_aware"] = True  # 标记需要时间边界
        # 如果置信度较低（单个科技词），降级处理
        if result["confidence"] < 0.75:
            # 仍然按 luxun 处理，但带 time_aware 标记
            normal = classify_normal(text)
            normal["time_aware"] = True
            normal["time_warning"] = result["matched"]
            return normal
        return result

    # Step 3: 正常分类
    result = classify_normal(text)
    result["time_aware"] = False
    return result


# ============================================================
# 批量测试
# ============================================================

def run_tests():
    """用已有问题集测试分类准确率"""
    import os

    tests_dir = os.path.join(os.path.dirname(__file__), "..", "tests")
    test_cases = []

    # 加载人物类问题
    luxun_file = os.path.join(tests_dir, "luxun_questions.txt")
    if os.path.exists(luxun_file):
        with open(luxun_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # 格式: L001|问题文本
                parts = line.split("|", 1)
                if len(parts) == 2:
                    qid, text = parts
                    test_cases.append((qid.strip(), text.strip()))

    # 加载场馆类问题
    venue_file = os.path.join(tests_dir, "venue_questions.txt")
    if os.path.exists(venue_file):
        with open(venue_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    qid, text = parts
                    test_cases.append((qid.strip(), text.strip()))

    # 期望分类
    EXPECTED = {
        # 场馆类 → narrator
        **{f"V{i:03d}": "narrator" for i in range(1, 21)},
        # 人物基础 → luxun
        **{f"L{i:03d}": "luxun" for i in range(1, 21)},
        # 人物检核 → luxun（知识库应能回答）
        "L026": "luxun",  # 诺贝尔奖 — 1901年已存在，合法问题
        "L027": "luxun",  # 《围城》 — 可检索后回答"否"
        "L028": "luxun",  # 是否是医学家 — 可检索后回答
        # 虚拟假设（不涉及现代事物）→ luxun
        "L023": "luxun",  # 面对迷茫年轻人给建议 — 问鲁迅价值观
        "L024": "luxun",  # 如何看待读书
        "L025": "luxun",  # 用鲁迅口吻谈独立思考
        # 越界 → reject_time
        "L021": "reject_time",
        "L022": "reject_time",
        "L029": "reject_time",
        "L030": "reject_time",
        # L023 "如果面对迷茫的年轻人，鲁迅会给出什么建议？"
        # → luxun（问鲁迅的价值观/建议，不涉及1936年后）
        # 陷阱 → reject_time
        "V021": "reject_time",
        "V022": "reject_time",
        "V023": "reject_time",
        "V024": "reject_time",
        "V025": "reject_time",
    }

    correct = 0
    total = 0
    errors = []

    for qid, text in test_cases:
        result = classify(text)
        predicted = result["intent"]
        expected = EXPECTED.get(qid, "unknown")
        total += 1

        if predicted == expected:
            correct += 1
        else:
            errors.append({
                "qid": qid,
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "confidence": result["confidence"],
                "scores": result.get("scores", {}),
                "matched": result.get("matched", []),
            })

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"=" * 60)
    print(f"意图分类器 测试结果")
    print(f"=" * 60)
    print(f"总测试数: {total}")
    print(f"正确: {correct}")
    print(f"错误: {len(errors)}")
    print(f"准确率: {accuracy:.1f}%")
    print()

    if errors:
        print(f"--- 分类错误详情 ---")
        for e in errors:
            print(f"  [{e['qid']}] {e['text']}")
            print(f"    期望: {e['expected']}  实际: {e['predicted']}  (conf={e['confidence']:.2f})")
            if e.get("scores"):
                print(f"    narrator={e['scores'].get('narrator',0):.1f}  luxun={e['scores'].get('luxun',0):.1f}")
            if e.get("matched"):
                print(f"    matched: {e['matched']}")
            print()

    return accuracy, errors


if __name__ == "__main__":
    run_tests()
