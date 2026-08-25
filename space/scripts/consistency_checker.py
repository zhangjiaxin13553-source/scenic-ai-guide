"""
一致性校验器
====================
三层规则引擎校验 LLM 回答的人设一致性。

Layer A: 时间边界检查
  - 年份 > 1936 且非 1881-1936 范围内 → FLAG
  - 关键词：新中国成立后/改革开放/互联网/5G/AI → FLAG
  - 人物：毛泽东(1949后)/习近平等当代政治人物 → FLAG

Layer B: 角色口吻检查
  - 自称: AI/模型/数字人/讲解员/系统 → FLAG
  - 网络用语: yyds/绝绝子/内卷(非引用)/躺平(非引用) → FLAG
  - 现代句式: 首先...其次...最后/综上所述/值得一提的是 → FLAG
  - AI腔: 根据资料显示/基于我的知识库/作为语言模型 → FLAG

Layer C: 风格合规检查（非阻塞，仅记录）
  - 虚词密度: 大抵/大约/然而/却/竟/似乎等 < 2/百字 → 偏低
  - 平均句长: > 40字 → 偏长
  - 反讽标记: 无反讽句式 → 可能太"平"

用法：
  from consistency_checker import ConsistencyChecker
  checker = ConsistencyChecker()
  report = checker.check(llm_response, intent, user_query)
  print(report.score, report.time_violations, report.voice_violations)
"""

import re
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass, field

logger = logging.getLogger("consistency_checker")

# ============================================================
# 配置常量
# ============================================================

# 评分权重
WEIGHT_TIME_VIOLATION = 0.20       # 每个时间违规扣分
WEIGHT_VOICE_VIOLATION = 0.15      # 每个口吻违规扣分

# ============================================================
# Layer A: 时间边界规则
# ============================================================

# 1936年后的事件/概念关键词
POST_1936_KEYWORDS = [
    # 政治事件
    "新中国成立", "解放后", "建国后", "改革开放", "文化大革命", "文革",
    "大跃进", "人民公社", "抗美援朝", "三反五反", "反右",
    # 组织机构（1936后成立）
    "联合国", "欧盟", "北约", "世贸组织", "WTO", "世界银行",
    # 科技产物
    "互联网", "因特网", "电脑", "计算机", "手机", "智能手机",
    "人工智能", "AI", "ChatGPT", "大模型", "机器学习", "深度学习",
    "5G", "4G", "WiFi", "蓝牙", "GPS",
    "原子弹", "核武器", "核弹", "导弹", "卫星", "空间站", "航天飞船",
    "高铁", "动车", "地铁", "磁悬浮",
    "电视", "电视机", "彩色电视", "液晶",
    "二维码", "扫码", "移动支付", "支付宝", "微信支付",
    # 社交媒体/平台
    "微信", "微博", "抖音", "B站", "bilibili", "小红书", "快手",
    "知乎", "贴吧", "QQ", "社交媒体", "短视频",
    # 现代概念
    "996", "内卷", "躺平", "佛系", "内耗", "PUA",
    "奥斯卡", "诺贝尔和平奖", "格莱美",
    "奥运会", "冬奥会", "亚运会",
    # 现代品牌
    "淘宝", "天猫", "京东", "拼多多", "美团", "饿了么",
    "滴滴", "共享单车", "网购", "外卖",
    # 现代娱乐
    "综艺", "真人秀", "直播", "网红", "电竞", "动漫展",
    "科幻电影", "漫威", "DC", "好莱坞大片",
    # 现代医疗
    "抗生素", "疫苗", "CT", "核磁共振", "B超", "化疗",
]

# 1936年后活跃的人物
POST_1936_PERSONS = [
    "莫言", "村上春树", "余华", "贾平凹", "王安忆", "金庸",
    "钱学森", "邓稼先", "袁隆平", "屠呦呦",
    "乔布斯", "比尔盖茨", "马斯克", "马云", "马化腾", "任正非",
    # 注：毛泽东、周恩来等在1936年前已活跃，但1949年后的角色属于越界
]

# 当代政治人物（鲁迅不应评价）
CONTEMPORARY_POLITICAL = [
    "习近平", "毛泽东时代", "周恩来总理", "邓小平",
    "江泽民", "胡锦涛", "温家宝", "李克强",
]

# 1936年后的具体年份
RE_POST_1936_YEAR = re.compile(r'(?:公元|公历)?(19[4-9]\d|20[0-2]\d)年')

# 时间边界 — 场景模式（整个上下文属于时间越界）
TIME_CLASH_CONTEXT = [
    re.compile(r'(?:如果|假如|假设).*(?:鲁迅|你).*(?:活在|生活在|来到|穿越到).*(?:今天|现代|当代|现在|当下)'),
    re.compile(r'(?:鲁迅|您|你).*(?:怎么看|怎么评价|如何看待).*(?:现代|当代|当今|当下|现在|年轻人)'),
    re.compile(r'(?:请|让|要求).*(?:鲁迅|你).*(?:评价|看待|评论).*(?:现代|当代|当今).*'),
]

# 鲁迅时代已有的事物 — 不应误判
PRE_1936_SAFE = [
    "火车", "轮船", "汽车", "飞机",  # 20世纪初已存在
    "电报", "电话", "留声机", "唱片", "无线电", "广播",
    "电影", "无声电影", "黑白电影",
    "照相机", "摄影",
    "电灯", "电话",
    "自行车", "黄包车", "马车",
    "自来水", "煤气",
    "报纸", "杂志", "刊物", "出版",
    "北京大学", "燕京大学", "清华大学",
    "新青年",
    "五四", "五四运动", "新文化运动",
]

# ============================================================
# 拒绝语境豁免 (Refusal Context Exemption)
# ============================================================
# 当回答是"拒绝式回答"（坦承不知 / 越界拒答）时，为解释拒绝而
# 必然提及的现代词/人物/年份，不应计为时间违规。否则"正确拒绝"
# 会被误判为"时间穿越"，产生假 FAIL（该误判风险此前已预警）。

# 回答级拒绝标记：出现任一即判定整段为拒绝式回答
REFUSAL_MARKERS = [
    "不知道", "不晓得", "不知晓", "不懂", "不了解", "不清楚", "不熟悉",
    "未曾听说", "未曾见过", "未曾听闻", "未曾经历", "不认识", "没听说",
    "不能回答", "无法回答", "无从回答", "难以回答", "无可奉告", "无从谈起",
    "恐怕难以", "不便回答", "不便妄加", "不能妄加", "不敢妄议", "不便置评",
    "不是我所能", "非我所知", "说不上来", "想不起来",
    "属于另一个时代", "另一个时代", "身后", "生前", "死后", "超出我",
    "在我的记忆之外", "记忆之外",
    "资料有限", "资料缺乏", "所据有限", "我手头",
]

# 转述用户提问的表述（回答在引用用户原话，而非自己断言）
QUOTE_FRAMING_MARKERS = [
    "你问的", "你问起", "你说的", "你所问", "你所提", "你所言",
    "提到的", "所提到的", "所谓", "问题是", "问及", "问的是",
]


def _is_refusal(text: str) -> bool:
    """判断整段回答是否为拒绝式回答（坦承不知 / 越界拒答）。"""
    return any(m in text for m in REFUSAL_MARKERS)


def _quote_framed(text: str, idx: int, window: int = 25) -> bool:
    """关键词 idx 之前是否在转述用户提问（你问的 / 你说的 …）。"""
    pre = text[max(0, idx - window): idx]
    return any(m in pre for m in QUOTE_FRAMING_MARKERS)


def _near_refusal(text: str, idx: int, window: int = 50) -> bool:
    """关键词 idx 前后 window 字符内是否出现拒绝标记。"""
    seg = text[max(0, idx - window): min(len(text), idx + window)]
    return any(m in seg for m in REFUSAL_MARKERS)


def _exempt_mention(text: str, idx: int) -> bool:
    """拒绝语境豁免：转述用户提问，或作为拒绝解释的一部分被提及。"""
    return _quote_framed(text, idx) or _near_refusal(text, idx)


def _check_time_boundary(text: str, intent: str) -> List[dict]:
    """
    Layer A: 时间边界检查。

    Returns:
        [{rule: str, matched: str, severity: "high"|"medium"|"low"}, ...]
    """
    violations = []

    # 1. 检查1936年后具体年份
    years = RE_POST_1936_YEAR.findall(text)
    for yr in years:
        yr_int = int(yr)
        if yr_int > 1936:
            idx = text.find(f"{yr}年")
            if _exempt_mention(text, idx):
                continue  # 拒绝语境豁免：转述用户提问/解释拒绝时提及的年份
            violations.append({
                "layer": "time",
                "rule": "post_1936_year",
                "matched": f"{yr}年",
                "severity": "high" if yr_int > 1950 else "medium",
                "detail": f"鲁迅1936年逝世，不应知晓{yr}年的事件",
            })

    # 2. 检查1936年后关键词
    for kw in POST_1936_KEYWORDS:
        if kw in text:
            idx = text.find(kw)
            if _exempt_mention(text, idx):
                continue  # 拒绝语境豁免：转述用户提问/解释拒绝时提及现代词
            violations.append({
                "layer": "time",
                "rule": "post_1936_keyword",
                "matched": kw,
                "severity": "high",
                "detail": f"'{kw}' 是1936年后的事物/概念",
            })

    # 3. 检查当代人物
    for person in POST_1936_PERSONS:
        if person in text:
            idx = text.find(person)
            if _exempt_mention(text, idx):
                continue  # 拒绝语境豁免：拒绝/坦承不知时提及的当代人物
            violations.append({
                "layer": "time",
                "rule": "post_1936_person",
                "matched": person,
                "severity": "high",
                "detail": f"'{person}' 是1936年后的作家/人物",
            })

    # 4. 检查当代政治人物
    for person in CONTEMPORARY_POLITICAL:
        if person in text:
            idx = text.find(person)
            if _exempt_mention(text, idx):
                continue  # 拒绝语境豁免：拒绝评价当代政治人物是正确行为
            violations.append({
                "layer": "time",
                "rule": "contemporary_political",
                "matched": person,
                "severity": "high",
                "detail": f"鲁迅不应评价当代政治人物 '{person}'",
            })

    # 5. 检查时间穿越假设场景
    for pat in TIME_CLASH_CONTEXT:
        m = pat.search(text)
        if m:
            # 检查整段是否在"拒绝回答"——如果是拒绝，不算违规
            matched_text = m.group()
            pos = text.find(matched_text)
            after_context = text[pos:pos + 150]
            if any(reject in after_context for reject in [
                "我不知道", "我不了解", "不能回答", "无法回答",
                "恐怕难以", "不是我所能", "属于另一个时代",
            ]):
                continue
            violations.append({
                "layer": "time",
                "rule": "time_clash_context",
                "matched": matched_text[:60],
                "severity": "high",
                "detail": "回答参与了时间穿越假设",
            })

    return violations


# ============================================================
# Layer B: 角色口吻规则
# ============================================================

# AI 自称词（鲁迅/讲解员不应出现）
AI_SELF_REFERENCES = [
    "作为AI", "作为人工智能", "作为一个AI",
    "我是AI", "我是一个AI", "我是一个人工智能",
    "作为语言模型", "作为一个语言模型",
    "我是语言模型", "我是大模型",
    "作为数字人", "我是数字人", "作为一个数字人",
    "基于我的知识库", "根据我的训练数据",
    "根据资料显示", "根据数据库", "据我的资料库",
    "根据现有资料", "根据我所掌握的资料",
    "我的数据库", "我的知识库",
    "系统提示", "根据系统设定",
    "我被设定为", "我的设定是",
]

# 现代网络用语（鲁迅不可能使用的表达）
MODERN_SLANG = [
    "yyds", "绝绝子", "破防了", "emo了", "芭比Q",
    "栓Q", "真的会谢", "大无语", "无语子",
    "摆烂", "摸鱼", "划水",
    "get到", "get不到", "get到了吗",
    "CPU", "KTV", "PPT", "DNA动了",
    "社恐", "社牛", "社死", "社交牛逼症",
    "显眼包", "嘴替", "互联网嘴替",
    "泰酷辣", "家人们", "谁懂啊",
]

# 现代句式/模板（AI生成痕迹）
MODERN_SENTENCE_PATTERNS = [
    (re.compile(r'首先[，,].*其次[，,].*最后[，,]'), "现代议论文句式"),
    (re.compile(r'综上所述[，,，。]'), "现代议论文句式"),
    (re.compile(r'值得一提的是[，,]'), "现代议论文句式"),
    (re.compile(r'总的来说[，,]'), "现代总结句式"),
    (re.compile(r'总体而言[，,]'), "现代总结句式"),
    (re.compile(r'从某种(?:意义|程度)上说'), "现代论述句式"),
    (re.compile(r'值得(?:注意|关注|思考|警惕)的是'), "现代论述句式"),
    (re.compile(r'不容(?:忽视|乐观|小觑)'), "现代评价句式"),
    (re.compile(r'堪称[一二三四五六七八九十\w]+'), "现代评价句式"),
    (re.compile(r'可谓[一二三四五六七八九十\w]+'), "现代评价句式"),
]

# AI 模板式结尾
AI_TEMPLATE_ENDINGS = [
    re.compile(r'(?:希望|但愿|盼|祝愿).*(?:对您|对你|能|可以).*(?:帮助|有用|启发|收获)'),
    re.compile(r'(?:如有|如果还有|若有).*(?:问题|疑问|需要).*(?:欢迎|请随时|可以继续).*'),
    re.compile(r'(?:以上|这些).*(?:就是|便是|即是).*(?:关于|我对).*(?:回答|介绍|说明|分享)'),
]


def _check_voice(text: str, intent: str) -> List[dict]:
    """
    Layer B: 角色口吻检查。

    Returns:
        [{layer: "voice", rule: str, matched: str, severity: str, detail: str}, ...]
    """
    violations = []

    # 1. AI 自称
    for ref in AI_SELF_REFERENCES:
        if ref in text:
            violations.append({
                "layer": "voice",
                "rule": "ai_self_reference",
                "matched": ref,
                "severity": "high",
                "detail": "鲁迅/讲解员不应自称AI、模型或数字人",
            })

    # 2. 网络用语
    for slang in MODERN_SLANG:
        if slang in text:
            violations.append({
                "layer": "voice",
                "rule": "modern_slang",
                "matched": slang,
                "severity": "high",
                "detail": f"网络用语'{slang}'不符合鲁迅时代语言",
            })

    # 3. 现代句式
    for pat, label in MODERN_SENTENCE_PATTERNS:
        m = pat.search(text)
        if m:
            violations.append({
                "layer": "voice",
                "rule": "modern_sentence_pattern",
                "matched": m.group()[:40],
                "severity": "medium",
                "detail": f"{label}，有AI生成痕迹",
            })

    # 4. AI 模板式结尾
    for pat in AI_TEMPLATE_ENDINGS:
        m = pat.search(text)
        if m:
            violations.append({
                "layer": "voice",
                "rule": "ai_template_ending",
                "matched": m.group()[:50],
                "severity": "medium",
                "detail": "模板化结尾有AI痕迹，不够自然",
            })

    # 5. 讲解员模式特有检查
    if intent == "narrator":
        # 讲解员不应使用第一人称扮演鲁迅
        narrator_bad_patterns = [
            (re.compile(r'我(?:以为|觉得|认为|想|看).*'), "讲解员不应以第一人称表达观点"),
            (re.compile(r'我的(?:作品|文章|小说|书|时代|一生)'), "讲解员不应以鲁迅第一人称说话"),
        ]
        for pat, detail in narrator_bad_patterns:
            m = pat.search(text)
            if m:
                violations.append({
                    "layer": "voice",
                    "rule": "narrator_first_person",
                    "matched": m.group()[:40],
                    "severity": "medium",
                    "detail": detail,
                })

    # 6. 数字人模式特有检查
    if intent in ("luxun", "ambiguous"):
        # "鲁迅先生"称呼自己（鲁迅不应自称"鲁迅先生"）
        if "鲁迅先生" in text:
            idx = text.find("鲁迅先生")
            context_pre = text[max(0, idx - 10):idx]
            if "我是" not in context_pre and "我就是" not in context_pre:
                # 可能是引用他人称呼，检查上下文
                if "自称" not in text[max(0, idx - 30):idx + 20]:
                    pass  # 这个不强行判违规，取决于上下文

    return violations


# ============================================================
# Layer C: 风格合规检查
# ============================================================

# 鲁迅特征虚词
LUXUN_FUNCTION_WORDS = [
    "大抵", "大约", "也许", "恐怕", "或许", "似乎", "仿佛",
    "然而", "但是", "但", "却", "竟", "倒", "反倒",
    "倘", "倘若", "若是", "假如", "如果",
    "便是", "即是", "正是", "还是",
    "自然", "当然", "固然", "果然",
    "几乎", "简直", "索性", "干脆",
    "无非", "不过", "总算", "终究",
    "向来", "一向", "素来", "从来",
    "一面", "一面……一面", "一边……一边",
]

# 鲁迅特征句式
LUXUN_STYLE_PATTERNS = [
    re.compile(r'.*(?:大抵|大约|也许|恐怕).*(?:是|有|可以|便是)'),
    re.compile(r'.*(?:然而|但是|但|却).*'),
    re.compile(r'.*我想[，,].*'),
    re.compile(r'.*我以为[，,].*'),
    re.compile(r'.*这(?:大约|大抵|也许|恐怕).*'),
    re.compile(r'.*究竟.*是怎么回事'),
    re.compile(r'.*算不得.*'),
    re.compile(r'.*也未必.*'),
    re.compile(r'.*倒也不.*'),
    re.compile(r'.*想来.*'),
    re.compile(r'.*说不定.*'),
]

# 反讽标记
IRONY_PATTERNS = [
    re.compile(r'.*(?:自然是|自然是了|自然如此).*'),
    re.compile(r'.*(?:原是|原是如此).*'),
    re.compile(r'.*(?:好得很|妙得很|有趣得很).*'),
    re.compile(r'.*倒(?:是|也).*有趣.*'),
    re.compile(r'.*(?:岂不|岂非|难道).*'),
    re.compile(r'.*(?:真可谓|真真是).*'),
]


def _check_style(text: str, intent: str) -> dict:
    """
    Layer C: 风格合规检查（非阻塞，仅记录）。

    Returns:
        {metrics: {...}, notes: [...]}
    """
    if intent == "narrator":
        # 讲解模式不检查风格
        return {"metrics": {}, "notes": ["讲解模式不适用风格检查"]}

    # 清理文本
    clean = text.replace(" ", "").replace("\n", "")

    # 计算虚词密度
    function_word_count = 0
    for fw in LUXUN_FUNCTION_WORDS:
        function_word_count += clean.count(fw)
    char_count = len(clean)
    density = function_word_count / max(char_count, 1) * 100  # 每百字

    # 计算平均句长
    sentences = re.split(r'[。！？；\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    total_chars = sum(len(s) for s in sentences)
    avg_sentence_len = total_chars / max(len(sentences), 1)

    # 检测反讽
    irony_count = sum(1 for pat in IRONY_PATTERNS if pat.search(text))

    # 风格句式匹配
    style_matches = sum(1 for pat in LUXUN_STYLE_PATTERNS if pat.search(text))

    # 诊断意见
    notes = []
    metrics = {
        "function_word_density": round(density, 2),
        "avg_sentence_length": round(avg_sentence_len, 1),
        "sentence_count": len(sentences),
        "irony_markers": irony_count,
        "style_patterns_matched": style_matches,
    }

    if density < 2.0:
        notes.append(f"虚词密度偏低 ({density:.1f}/百字)，建议增加'大抵/大约/然而/却'等鲁迅常用虚词")
    if avg_sentence_len > 40:
        notes.append(f"平均句长偏长 ({avg_sentence_len:.0f}字)，鲁迅风格应控制在30字以内")
    if irony_count == 0:
        notes.append("未检测到反讽标记，回答可能太'平'，缺少鲁迅式的冷峻/反讽")
    if style_matches < 2:
        notes.append("鲁迅特征句式匹配较少，建议增加'我想...'、'这大约...'等句式")

    return {"metrics": metrics, "notes": notes}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ConsistencyReport:
    """一致性校验报告"""
    score: float                              # 0.0 ~ 1.0
    time_violations: List[dict] = field(default_factory=list)
    voice_violations: List[dict] = field(default_factory=list)
    style_notes: List[str] = field(default_factory=list)
    style_metrics: dict = field(default_factory=dict)
    summary: str = ""


# ============================================================
# ConsistencyChecker
# ============================================================

class ConsistencyChecker:
    """
    一致性校验器。

    三层规则引擎：
      Layer A — 时间边界：检测 post-1936 的日期/事件/概念/人物
      Layer B — 角色口吻：检测 AI 腔/网络用语/现代句式/模板结尾
      Layer C — 风格合规：虚词密度/句长/反讽（非阻塞，仅记录）

    使用方式：
        checker = ConsistencyChecker()
        report = checker.check(llm_response, intent="luxun", user_query="...")
        print(report.score, report.time_violations)
    """

    def __init__(
        self,
        weight_time: float = WEIGHT_TIME_VIOLATION,
        weight_voice: float = WEIGHT_VOICE_VIOLATION,
    ):
        self.weight_time = weight_time
        self.weight_voice = weight_voice

    def check(
        self,
        llm_response: str,
        intent: str = "luxun",
        user_query: str = "",
    ) -> ConsistencyReport:
        """
        对 LLM 回答执行一致性检查。

        Args:
            llm_response: LLM 生成的回答文本
            intent:       意图分类结果 (narrator/luxun/ambiguous)
            user_query:   用户原始问题（可选）

        Returns:
            ConsistencyReport
        """
        if not llm_response or not llm_response.strip():
            return ConsistencyReport(
                score=1.0,
                summary="回答为空",
            )

        # Layer A: 时间边界
        time_violations = _check_time_boundary(llm_response, intent)

        # Layer B: 角色口吻
        voice_violations = _check_voice(llm_response, intent)

        # Layer C: 风格合规
        style_result = _check_style(llm_response, intent)

        # 计算评分
        score = 1.0
        high_time = sum(1 for v in time_violations if v["severity"] == "high")
        med_time = sum(1 for v in time_violations if v["severity"] == "medium")
        high_voice = sum(1 for v in voice_violations if v["severity"] == "high")
        med_voice = sum(1 for v in voice_violations if v["severity"] == "medium")

        # 阈值微调：讲解模式宽松处理（可能涉及现代纪念馆运营），时间违规扣分减半
        time_scale = 0.5 if intent == "narrator" else 1.0
        score -= high_time * self.weight_time * 1.5 * time_scale
        score -= med_time * self.weight_time * 1.0 * time_scale
        score -= high_voice * self.weight_voice * 1.5
        score -= med_voice * self.weight_voice * 1.0
        score = max(0.0, min(1.0, score))

        # 汇总
        parts = []
        if time_violations:
            parts.append(f"⏰ {len(time_violations)} 处时间违规")
        if voice_violations:
            parts.append(f"🎭 {len(voice_violations)} 处口吻违规")
        if style_result["notes"]:
            parts.append(f"✏ {len(style_result['notes'])} 条风格建议")
        if not parts:
            parts.append("✅ 一致性检查通过")

        return ConsistencyReport(
            score=score,
            time_violations=time_violations,
            voice_violations=voice_violations,
            style_notes=style_result.get("notes", []),
            style_metrics=style_result.get("metrics", {}),
            summary="；".join(parts),
        )


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-5s | %(message)s",
    )

    print("=" * 60)
    print("  一致性校验器 自测")
    print("=" * 60)

    checker = ConsistencyChecker()

    test_cases = [
        # Case 1: 正常鲁迅口吻回答
        (
            "我想，青年总该是有希望的。然而现在的世道，"
            "却也未必能让他们顺着自己的意思生长。这大约是我的一点感慨罢。",
            "luxun",
            "正常鲁迅口吻",
        ),
        # Case 2: 时间穿越
        (
            "我觉得改革开放确实给中国带来了很大变化，"
            "互联网的发展也改变了人们的生活方式。",
            "luxun",
            "时间穿越+现代概念",
        ),
        # Case 3: AI 腔
        (
            "根据我的知识库显示，鲁迅生于1881年。"
            "首先，他是中国现代文学的奠基人。其次，他的作品影响深远。"
            "最后，综上所述，鲁迅是一位伟大的作家。"
            "希望这些信息对您有所帮助！",
            "luxun",
            "AI腔严重",
        ),
        # Case 4: 网络用语
        (
            "这个问题我get不到。现在的年轻人真是绝绝子，"
            "鲁迅要是看到他们躺平的样子，估计会破防了吧。",
            "luxun",
            "网络用语泛滥",
        ),
        # Case 5: 讲解模式正常
        (
            "广州鲁迅纪念馆位于广州市越秀区文明路215号，"
            "原为中山大学钟楼。纪念馆展示了鲁迅1927年在广州期间的生活与创作。",
            "narrator",
            "讲解模式正常",
        ),
    ]

    for i, (response, intent, desc) in enumerate(test_cases, 1):
        print(f"\n--- Case {i}: {desc} (intent={intent}) ---")
        report = checker.check(response, intent=intent)

        print(f"  评分: {report.score:.2f}")
        print(f"  摘要: {report.summary}")

        if report.time_violations:
            print(f"  时间违规 ({len(report.time_violations)}):")
            for v in report.time_violations:
                print(f"    [{v['severity']}] {v['rule']}: {v['matched'][:50]}")

        if report.voice_violations:
            print(f"  口吻违规 ({len(report.voice_violations)}):")
            for v in report.voice_violations:
                print(f"    [{v['severity']}] {v['rule']}: {v['matched'][:50]}")

        if report.style_notes:
            print(f"  风格建议 ({len(report.style_notes)}):")
            for n in report.style_notes:
                print(f"    - {n}")
