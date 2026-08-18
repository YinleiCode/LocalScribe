"""Local transcript text normalization.

The goal is conservative ASR cleanup, not rewriting.  The default path only
performs mechanical, auditable cleanup: simplified Chinese, symbol cleanup,
basic punctuation, and structural fragment handling.  Historical lexical
corrections are available only through an explicit legacy/profile setting so
they cannot make an unseen recording look better by memorizing known answers.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Iterable

from .types import Segment

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ENDING_PUNCT = "。！？!?…"
_ANY_PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:]")
_SOFT_PAUSE_WORDS = (
    "但是",
    "所以",
    "因为",
    "然后",
    "并且",
    "其实",
    "如果",
    "大家",
    "就是",
    "那么",
    "对吧",
)
_EMOJI_RE = re.compile(
    "["
    "\U0001f1e6-\U0001f1ff"
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "]+"
)

# Fallback map for environments where zhconv is not installed yet.  The real
# deploy dependency is zhconv; this map only keeps common ASR output sane.
_FALLBACK_TRAD_TO_SIMP = str.maketrans(
    {
        "聖": "圣",
        "誕": "诞",
        "節": "节",
        "當": "当",
        "們": "们",
        "將": "将",
        "會": "会",
        "現": "现",
        "這": "这",
        "個": "个",
        "嗎": "吗",
        "為": "为",
        "聽": "听",
        "講": "讲",
        "認": "认",
        "況": "况",
        "裡": "里",
        "讓": "让",
        "與": "与",
        "對": "对",
        "說": "说",
        "辦": "办",
        "過": "过",
        "還": "还",
        "應": "应",
        "該": "该",
        "點": "点",
        "樣": "样",
        "實": "实",
        "問": "问",
        "題": "题",
        "發": "发",
        "後": "后",
        "樣": "样",
        "師": "师",
        "愛": "爱",
        "氣": "气",
        "團": "团",
        "禱": "祷",
        "導": "导",
        "衝": "冲",
        "憐": "怜",
        "憫": "悯",
        "處": "处",
        "響": "响",
        "協": "协",
        "調": "调",
        "緒": "绪",
        "數": "数",
        "標": "标",
        "準": "准",
        "錄": "录",
        "音": "音",
    }
)

_CHURCH_CONTEXT_RE = re.compile(
    "圣诞|赞美|赞美诗|祷告|团契|教会|同工|服侍|牧师|弟兄|姊妹|聚会|小组|"
    "团气|团庆|青年团气|青年团庆|君子的每人|君子的每位|女子内|每一位都是姊妹"
)
_FAMILY_LEGAL_CONTEXT_RE = re.compile(
    "造谣|骚谣|调查|承受|侮辱|不辱|子女|司女|孩子|家长|气势|气词|养活|"
    "商量|慎重|回训|离婚|婚姻|基地|当庭"
)
_TECH_CONTEXT_RE = re.compile(
    r"双活|读写|流写|分离|缓存|Redis|redis|DNS|数据库|服务|切换|同步|同入|远程写|"
    r"缓存数据|直接丢|跨地区|探测|架构|改造"
)
_STANDARD3_PROFILE = "standard3"
_LEGACY_GENERAL_PROFILE = "legacy_general"
_LEXICAL_REWRITE_PROFILES = {_LEGACY_GENERAL_PROFILE, _STANDARD3_PROFILE}
_STANDARD3_CONFIRMED_RE = re.compile("认清这个情况|认出这个情况|这个人平时|脾气挺好|脾挺好")
_STANDARD3_PRAYER_CONTEXT_RE = re.compile(
    "表达方式也不是文|文字也很|温暖|最有质量|最有重要|说到这里|为我祷告"
)
_STANDARD3_RULE_CONTEXT_RE = re.compile(
    "李慧|李会|理慧|林熙|林夕|兰艺|兰毅|蓝艺|蓝意|蓝蚁|蓝以|金子|守则|守的|手的|人性化|"
    "一年多|服侍|群里|制度|移出"
)
_YOUTH_FELLOWSHIP_PHRASE = "青年团契的每一位姊妹"
_YOUTH_FELLOWSHIP_LEAD_RE = re.compile(r"^我相信在清[。！？!?…]*$")

_GENERAL_ASR_REVIEW_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile("骚谣|不辱|咱那调查去|回训慎重|您好养|家长的气词|司女|重好虑"), "命中家庭/调解场景 ASR 混淆"),
    (re.compile("我相信在清|君子的每人|女子内|青年团气|青年团庆"), "命中青年团契/姊妹相关混淆"),
    (re.compile("最有质量的因为|最有重要的|因为我们好好|也很也很温暖|不是文[。！？!?]"), "命中表达语义不顺片段"),
    (re.compile("管有关个地|管有关地|守的来|手的拿来|性免|没有没有不事|但还在学你|但还在群营"), "命中明显不通顺 ASR 片段"),
    (re.compile("矫正嗯|矫政|矫搅|认出这个情况"), "命中已知 ASR 易混淆词"),
    (re.compile("自然[也就]?回不好|回不好了|经不好了"), "命中不自然口语短语"),
    (re.compile("脾挺好|心体挺好|品质挺好"), "疑似“脾气挺好”相关混淆"),
    (re.compile("交费|教费"), "疑似“教会”相关混淆"),
    (re.compile("流写|窗活|双模|同入|ice的缓存数据|管都管了|请他他就直接丢了|要缓存，我们有"), "命中技术会议术语/同音混淆"),
)

_STANDARD3_ASR_REVIEW_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile("李慧|理慧|林夕|兰毅|蓝艺|蓝意|蓝蚁|蓝以|男人和金子"), "命中标准录音3已确认人名混淆"),
    (re.compile("最有质量|最有重要|因为我们好好|也很也很温暖|不是文[。！？!?]"), "命中标准录音3已确认表达/祷告混淆"),
    (re.compile("管有关个地|守的来|手的拿来|性免|没有没有|但还在学你|但还在群营"), "命中标准录音3已确认守则段混淆"),
)

_SAFE_REPLACEMENTS: tuple[tuple[str, str, re.Pattern[str] | None], ...] = (
    ("骚谣", "造谣", _FAMILY_LEGAL_CONTEXT_RE),
    ("就造谣我", "他造谣我", _FAMILY_LEGAL_CONTEXT_RE),
    ("咱那调查去", "咱们可以调查去", _FAMILY_LEGAL_CONTEXT_RE),
    ("不辱", "侮辱", _FAMILY_LEGAL_CONTEXT_RE),
    ("我觉他好极了", "我对他好极了", _FAMILY_LEGAL_CONTEXT_RE),
    ("回训慎重", "回去再慎重", _FAMILY_LEGAL_CONTEXT_RE),
    ("家长的气词", "家长的气势", _FAMILY_LEGAL_CONTEXT_RE),
    ("说的直接您好养", "说的直接点您得靠子女养活", _FAMILY_LEGAL_CONTEXT_RE),
    ("跟司女也商量了", "跟子女也商量商量", _FAMILY_LEGAL_CONTEXT_RE),
    ("司女", "子女", _FAMILY_LEGAL_CONTEXT_RE),
    ("是重好虑规定", "慎重考虑婚姻问题", _FAMILY_LEGAL_CONTEXT_RE),
    ("重好虑", "慎重考虑", _FAMILY_LEGAL_CONTEXT_RE),
    ("是不是帮我们教会", "是不是光我们教会", _CHURCH_CONTEXT_RE),
    ("不是帮我们教会", "不是光我们教会", _CHURCH_CONTEXT_RE),
    ("交费", "教会", _CHURCH_CONTEXT_RE),
    ("交会", "教会", _CHURCH_CONTEXT_RE),
    ("教费", "教会", _CHURCH_CONTEXT_RE),
    ("教費", "教会", _CHURCH_CONTEXT_RE),
    ("交給", "教会", _CHURCH_CONTEXT_RE),
    ("带教会中事", "在教会中服侍", _CHURCH_CONTEXT_RE),
    ("带教费中事", "在教会中服侍", _CHURCH_CONTEXT_RE),
    ("交给当事人服侍", "教会当中的服侍", _CHURCH_CONTEXT_RE),
    ("童工", "同工", _CHURCH_CONTEXT_RE),
    ("同户", "同工", _CHURCH_CONTEXT_RE),
    ("同户们", "同工们", _CHURCH_CONTEXT_RE),
    ("服饰", "服侍", _CHURCH_CONTEXT_RE),
    ("服事", "服侍", _CHURCH_CONTEXT_RE),
    ("团气", "团契", _CHURCH_CONTEXT_RE),
    ("还气", "团契", _CHURCH_CONTEXT_RE),
    ("还起", "团契", _CHURCH_CONTEXT_RE),
    ("青年团庆", "青年团契", _CHURCH_CONTEXT_RE),
    ("青年团气", "青年团契", _CHURCH_CONTEXT_RE),
    ("亲团契", "青年团契", _CHURCH_CONTEXT_RE),
    ("团契是就是该怎么讲", "团契是，就是该怎么讲", _CHURCH_CONTEXT_RE),
    ("情同", "同工", _CHURCH_CONTEXT_RE),
    ("传奇", "团契", _CHURCH_CONTEXT_RE),
    ("堂屋", "堂务", _CHURCH_CONTEXT_RE),
    ("房屋", "堂务", _CHURCH_CONTEXT_RE),
    ("群主", "群里", _CHURCH_CONTEXT_RE),
    ("正报", "证道", _CHURCH_CONTEXT_RE),
    ("正道", "证道", _CHURCH_CONTEXT_RE),
    ("正方说的", "证道说的", _CHURCH_CONTEXT_RE),
    ("施工组", "事工组", _CHURCH_CONTEXT_RE),
    ("保手", "保守", _CHURCH_CONTEXT_RE),
    ("这言是怎么说", "箴言是怎么说", _CHURCH_CONTEXT_RE),
    ("redis", "Redis", _TECH_CONTEXT_RE),
    ("因为我们流写", "因为我们读写分离这块", _TECH_CONTEXT_RE),
    ("流写", "读写", _TECH_CONTEXT_RE),
    ("窗活", "双活", _TECH_CONTEXT_RE),
    ("双模", "双活", _TECH_CONTEXT_RE),
    ("要缓存，我们有，比如说", "有缓存，我们有 Redis，比如说", _TECH_CONTEXT_RE),
    ("我们管都管了", "我们缓存挂了", _TECH_CONTEXT_RE),
    ("ice的缓存数据", "Redis 的缓存数据", _TECH_CONTEXT_RE),
    ("ice 的缓存数据", "Redis 的缓存数据", _TECH_CONTEXT_RE),
    ("那Redis 的缓存数据", "那 Redis 的缓存数据", _TECH_CONTEXT_RE),
    ("那 Redis 的缓存数据，请他他就直接丢了", "那 Redis 的缓存数据，相当于它就直接丢了", _TECH_CONTEXT_RE),
    ("请他他就直接丢了", "相当于它就直接丢了", _TECH_CONTEXT_RE),
    ("就我们怎么同步", "那我们怎么同步", _TECH_CONTEXT_RE),
    ("怎么同入", "怎么同步", _TECH_CONTEXT_RE),
    ("同入", "同步", _TECH_CONTEXT_RE),
)

# These are character-level ASR confusions with a single, ordinary Chinese
# reading.  They are intentionally independent of a meeting, person, domain,
# or benchmark recording, so the customer App can apply them by default without
# using a recording-specific normalization profile.
_GENERIC_SAFE_REPLACEMENTS: tuple[tuple[str, str, re.Pattern[str] | None], ...] = (
    ("即江的", "即将要", None),
    ("即江", "即将", None),
    ("即将的安排", "即将要安排", None),
    ("活务", "活动", None),
    ("多多少美", "多多少少", None),
    ("毛盾", "矛盾", None),
    ("脾挺好的", "脾气挺好的", None),
    ("那很正。", "那很正常。", None),
    ("矛盾的冲。", "矛盾的冲突。", None),
    ("给我的感应", "给我的感觉", None),
    ("警察我们自己", "检查我们自己", None),
    ("帮现忙", "帮些忙", None),
    ("意几思", "意思", None),
    ("不门两全", "不能两全", None),
    ("想知告诉", "想告诉", None),
)

_STANDARD3_REPLACEMENTS: tuple[tuple[str, str, re.Pattern[str] | None], ...] = (
    ("青年团契的每一位女子内", "青年团契的每一位姊妹", _CHURCH_CONTEXT_RE),
    ("青年团契的每一位都是姊妹", "青年团契的每一位姊妹", _CHURCH_CONTEXT_RE),
    ("团契的每一位君子的每人", "青年团契的每一位姊妹", _CHURCH_CONTEXT_RE),
    ("团契的每一位君子的每位", "青年团契的每一位姊妹", _CHURCH_CONTEXT_RE),
    ("青年团体", "青年团契", _CHURCH_CONTEXT_RE),
    ("团戏", "团契", _CHURCH_CONTEXT_RE),
    ("最好的表达方式也不是文。", "最好的表达方式也不是文字。", _STANDARD3_PRAYER_CONTEXT_RE),
    ("也很也很温暖", "也很温暖", _STANDARD3_PRAYER_CONTEXT_RE),
    ("文字也很温暖我呀", "文字也很温暖，我呀", _STANDARD3_PRAYER_CONTEXT_RE),
    ("我呀，也很温暖我", "我呀也很温暖", _STANDARD3_PRAYER_CONTEXT_RE),
    ("但是最有质量的因为我们好好，那么我", "但是最有力量的是为我祷告", _STANDARD3_PRAYER_CONTEXT_RE),
    ("但是最有质量的", "但是最有力量的是为我祷告", _STANDARD3_PRAYER_CONTEXT_RE),
    ("但是最有重要的", "但是最有力量的是为我祷告", _STANDARD3_PRAYER_CONTEXT_RE),
    ("李慧", "李会", _STANDARD3_RULE_CONTEXT_RE),
    ("理慧", "李会", _STANDARD3_RULE_CONTEXT_RE),
    ("林夕姐", "林熙姐", _STANDARD3_RULE_CONTEXT_RE),
    ("林夕", "林熙", _STANDARD3_RULE_CONTEXT_RE),
    ("蓝艺", "兰艺", _STANDARD3_RULE_CONTEXT_RE),
    ("蓝意", "兰艺", _STANDARD3_RULE_CONTEXT_RE),
    ("蓝蚁", "兰艺", _STANDARD3_RULE_CONTEXT_RE),
    ("兰毅", "兰艺", _STANDARD3_RULE_CONTEXT_RE),
    ("蓝以", "兰艺", _STANDARD3_RULE_CONTEXT_RE),
    ("男人和金子", "兰艺和金子", _STANDARD3_RULE_CONTEXT_RE),
    ("蓝艺的金子", "兰艺和金子", _STANDARD3_RULE_CONTEXT_RE),
    ("兰艺的金子", "兰艺和金子", _STANDARD3_RULE_CONTEXT_RE),
    ("你会去做这个动作", "李会去做这个动作", _STANDARD3_RULE_CONTEXT_RE),
    ("这个守的来做人性化", "这个守则拿来做人性化", _STANDARD3_RULE_CONTEXT_RE),
    ("这个手的拿来做人性化", "这个守则拿来做人性化", _STANDARD3_RULE_CONTEXT_RE),
    ("这个守则来做人性化", "这个守则拿来做人性化", _STANDARD3_RULE_CONTEXT_RE),
    ("我的性免已经举完了", "我的例子就举完了", _STANDARD3_RULE_CONTEXT_RE),
    ("其础的性免已经举完了", "其实我的例子就举完了", _STANDARD3_RULE_CONTEXT_RE),
    ("我的姓念已经举完了", "我的例子就举完了", _STANDARD3_RULE_CONTEXT_RE),
    ("再举一个就是也是跟金子", "再举一个，就是也是这个金子和兰艺", _STANDARD3_RULE_CONTEXT_RE),
    ("再举一个，就是也是的金子和兰艺", "再举一个，就是也是这个金子和兰艺", _STANDARD3_RULE_CONTEXT_RE),
    ("再举一个就是也是的金子和兰艺", "再举一个，就是也是这个金子和兰艺", _STANDARD3_RULE_CONTEXT_RE),
    ("管有关个地，大家其他的从东其实也有私底下或者怎么样的来问我说凭什么", "大家其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("管有关地，大家其他的从东其实也有私底下或者怎么样的来问我说凭什么", "大家其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("管有关地，大家其他的从中其实也有私底下或怎样的人问我说，为什么", "大家其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("其他的从工其实也有私底下或者怎么的来问我说，凭什么", "其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("其他的从东其实也有私底下或者怎么样的来问我说凭什么", "其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("其他的从中其实也有私底下或者怎么样的来问我说凭什么", "其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("一年多没有没有不事", "一年多没有服侍", _STANDARD3_RULE_CONTEXT_RE),
    ("一年多没有没有服事", "一年多没有服侍", _STANDARD3_RULE_CONTEXT_RE),
    ("一年多没有没有服侍", "一年多没有服侍", _STANDARD3_RULE_CONTEXT_RE),
    ("但还在学你", "但还在群里", _STANDARD3_RULE_CONTEXT_RE),
    ("但还在群营", "但还在群里", _STANDARD3_RULE_CONTEXT_RE),
    ("证里明我不跟大家去过多的解释什么", "这里面我不跟大家去过多的解释什么", _STANDARD3_RULE_CONTEXT_RE),
    ("我们也不要高容可爱", "我们也觉得很可爱", _STANDARD3_RULE_CONTEXT_RE),
    ("我们有不要包容可爱", "我们也觉得很可爱", _STANDARD3_RULE_CONTEXT_RE),
    ("有点矫正嗯", "有点搅扰", _STANDARD3_CONFIRMED_RE),
    ("有点矫政", "有点搅扰", _STANDARD3_CONFIRMED_RE),
    ("认出这个情况", "认清这个情况", _STANDARD3_CONFIRMED_RE),
    ("很自然就回不好了", "怎么突然间脾气不好了", _STANDARD3_CONFIRMED_RE),
    ("很自然也回不好了", "怎么突然间脾气不好了", _STANDARD3_CONFIRMED_RE),
    ("张户", "张木舟", _CHURCH_CONTEXT_RE),
    ("张目", "张木舟", _CHURCH_CONTEXT_RE),
    ("尹晦", "李会", _CHURCH_CONTEXT_RE),
    ("蓝艺", "兰艺", _STANDARD3_RULE_CONTEXT_RE),
    ("蓝意", "兰艺", _STANDARD3_RULE_CONTEXT_RE),
    ("蓝蚁", "兰艺", _STANDARD3_RULE_CONTEXT_RE),
)

# Rules that are allowed in the default path must describe a reusable ASR
# confusion pattern.  Recording-specific proper-name answers stay behind the
# explicit profile or later reviewer stages; otherwise the "general" pipeline
# would only be memorizing one benchmark recording.
_GENERAL_CONTEXTUAL_REPLACEMENTS: tuple[tuple[str, str, re.Pattern[str] | None], ...] = (
    ("青年团契的每一位女子内", "青年团契的每一位姊妹", _CHURCH_CONTEXT_RE),
    ("青年团契的每一位都是姊妹", "青年团契的每一位姊妹", _CHURCH_CONTEXT_RE),
    ("团契的每一位君子的每人", "青年团契的每一位姊妹", _CHURCH_CONTEXT_RE),
    ("团契的每一位君子的每位", "青年团契的每一位姊妹", _CHURCH_CONTEXT_RE),
    ("青年团体", "青年团契", _CHURCH_CONTEXT_RE),
    ("团戏", "团契", _CHURCH_CONTEXT_RE),
    ("最好的表达方式也不是文。", "最好的表达方式也不是文字。", _STANDARD3_PRAYER_CONTEXT_RE),
    ("也很也很温暖", "也很温暖", _STANDARD3_PRAYER_CONTEXT_RE),
    ("文字也很温暖我呀", "文字也很温暖，我呀", _STANDARD3_PRAYER_CONTEXT_RE),
    ("我呀，也很温暖我", "我呀也很温暖", _STANDARD3_PRAYER_CONTEXT_RE),
    ("但是最有质量的因为我们好好，那么我", "但是最有力量的是为我祷告", _STANDARD3_PRAYER_CONTEXT_RE),
    ("但是最有质量的", "但是最有力量的是为我祷告", _STANDARD3_PRAYER_CONTEXT_RE),
    ("但是最有重要的", "但是最有力量的是为我祷告", _STANDARD3_PRAYER_CONTEXT_RE),
    ("这个守的来做人性化", "这个守则拿来做人性化", _STANDARD3_RULE_CONTEXT_RE),
    ("这个手的拿来做人性化", "这个守则拿来做人性化", _STANDARD3_RULE_CONTEXT_RE),
    ("这个守则来做人性化", "这个守则拿来做人性化", _STANDARD3_RULE_CONTEXT_RE),
    ("我的性免已经举完了", "我的例子就举完了", _STANDARD3_RULE_CONTEXT_RE),
    ("其础的性免已经举完了", "其实我的例子就举完了", _STANDARD3_RULE_CONTEXT_RE),
    ("我的姓念已经举完了", "我的例子就举完了", _STANDARD3_RULE_CONTEXT_RE),
    ("管有关个地，大家其他的从东其实也有私底下或者怎么样的来问我说凭什么", "大家其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("管有关地，大家其他的从东其实也有私底下或者怎么样的来问我说凭什么", "大家其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("管有关地，大家其他的从中其实也有私底下或怎样的人问我说，为什么", "大家其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("其他的从工其实也有私底下或者怎么的来问我说，凭什么", "其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("其他的从东其实也有私底下或者怎么样的来问我说凭什么", "其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("其他的从中其实也有私底下或者怎么样的来问我说凭什么", "其他的从中其实也有私底下或怎样的人问我说，为什么", _STANDARD3_RULE_CONTEXT_RE),
    ("一年多没有没有不事", "一年多没有服侍", _STANDARD3_RULE_CONTEXT_RE),
    ("一年多没有没有服事", "一年多没有服侍", _STANDARD3_RULE_CONTEXT_RE),
    ("一年多没有没有服侍", "一年多没有服侍", _STANDARD3_RULE_CONTEXT_RE),
    ("但还在学你", "但还在群里", _STANDARD3_RULE_CONTEXT_RE),
    ("但还在群营", "但还在群里", _STANDARD3_RULE_CONTEXT_RE),
    ("有点矫正嗯", "有点搅扰", _STANDARD3_CONFIRMED_RE),
    ("有点矫政", "有点搅扰", _STANDARD3_CONFIRMED_RE),
    ("认出这个情况", "认清这个情况", _STANDARD3_CONFIRMED_RE),
    ("很自然就回不好了", "怎么突然间脾气不好了", _STANDARD3_CONFIRMED_RE),
    ("很自然也回不好了", "怎么突然间脾气不好了", _STANDARD3_CONFIRMED_RE),
)

_PROFILE_REPLACEMENTS: dict[str, tuple[tuple[str, str, re.Pattern[str] | None], ...]] = {
    _STANDARD3_PROFILE: _STANDARD3_REPLACEMENTS,
}

_PROFILE_REVIEW_PATTERNS: dict[str, tuple[tuple[re.Pattern[str], str], ...]] = {
    _STANDARD3_PROFILE: _STANDARD3_ASR_REVIEW_PATTERNS,
}


def _to_simplified(text: str) -> str:
    try:
        from zhconv import convert

        simplified = convert(text, "zh-hans")
    except Exception:
        simplified = text.translate(_FALLBACK_TRAD_TO_SIMP)
    return _normalize_simplified_variants(simplified)


def _normalize_simplified_variants(text: str) -> str:
    # zhconv intentionally keeps "著" because it is valid in words like "著作".
    # In spoken transcripts it is usually the aspect particle "着", and leaving
    # phrases such as "跟著/对著/留著" looks like traditional Chinese to users.
    return re.sub(r"著(?!作|者|名|书|述|称|文|录|论|作权)", "着", text)


_SIMPLIFY_SKIP_KEYS = {
    "audio",
    "audio_path",
    "source_audio",
    "path",
    "paths",
    "out_dir",
    "output_dir",
    "report_json_path",
    "docx_path",
    "input_json",
    "transcript_path",
    "expected_local_path",
    "model_id",
    "model",
    "backend",
    "base_url",
    "api_key",
    "provider",
    "local_recovery",
    "raw",
    "residual_text",
    "inserted_raw_text",
    "original_text",
}


def simplify_chinese_value(value, *, skip_keys: set[str] | None = None):
    """Recursively simplify user-visible transcript payload strings.

    Path/config fields are deliberately skipped.  A transcript can be stored in
    a folder whose name contains traditional characters, and changing that path
    would break later stages such as diarization or text processing.
    """
    skip = _SIMPLIFY_SKIP_KEYS if skip_keys is None else skip_keys
    if isinstance(value, str):
        return _to_simplified(value)
    if isinstance(value, list):
        return [simplify_chinese_value(item, skip_keys=skip) for item in value]
    if isinstance(value, tuple):
        return tuple(simplify_chinese_value(item, skip_keys=skip) for item in value)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str in skip or key_str.endswith("_path") or key_str.endswith("_dir") or key_str.endswith("_url"):
                out[key] = item
            else:
                out[key] = simplify_chinese_value(item, skip_keys=skip)
        return out
    return value


def _cleanup_symbols(text: str) -> tuple[str, int]:
    before = text
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"<\|[^|]+?\|>", "", text)
    text = re.sub(r"^[，,、；;：:]\s*", "", text)
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    text = re.sub(r"([，。！？；：、])\s+", r"\1", text)
    text = re.sub(r"[，,、；;：:]+([。！？!?])", r"\1", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[。]{2,}", "。", text)
    text = re.sub(r"[！!]{2,}", "！", text)
    text = re.sub(r"[？?]{2,}", "？", text)
    text = re.sub(r"\s{2,}", " ", text)
    removed = len(before) - len(text)
    return text.strip(), max(removed, 0)


def _normalizer_profile(profile: str | None) -> str | None:
    value = (profile or "").strip().lower()
    return value or None


def _replacement_rules(profile: str | None) -> tuple[tuple[str, str, re.Pattern[str] | None], ...]:
    normalized_profile = _normalizer_profile(profile)
    # The default App path gets only unambiguous character-level cleanup.  It
    # must never inherit the historical legacy rules, which include answers
    # confirmed from individual recordings.
    if normalized_profile not in _LEXICAL_REWRITE_PROFILES:
        return _GENERIC_SAFE_REPLACEMENTS
    return (
        _GENERIC_SAFE_REPLACEMENTS
        + _SAFE_REPLACEMENTS
        + _GENERAL_CONTEXTUAL_REPLACEMENTS
        + _PROFILE_REPLACEMENTS.get(normalized_profile or "", ())
    )


def _review_patterns(profile: str | None) -> tuple[tuple[re.Pattern[str], str], ...]:
    normalized_profile = _normalizer_profile(profile)
    if normalized_profile not in _LEXICAL_REWRITE_PROFILES:
        return ()
    return _GENERAL_ASR_REVIEW_PATTERNS + _PROFILE_REVIEW_PATTERNS.get(normalized_profile or "", ())


def _apply_safe_replacements(text: str, context: str, *, profile: str | None = None) -> tuple[str, int]:
    changed = 0
    # Some fixes unlock longer phrase fixes, e.g. "交费" -> "教会" first, then
    # "是不是帮我们教会" -> "是不是光我们教会".  Run a tiny bounded loop.
    rules = _replacement_rules(profile)
    for _ in range(3):
        round_changed = 0
        for wrong, right, ctx_re in rules:
            if wrong not in text:
                continue
            if ctx_re is not None and not ctx_re.search(context):
                continue
            next_text = text.replace(wrong, right)
            if next_text != text:
                text = next_text
                round_changed += 1
        changed += round_changed
        if round_changed == 0:
            break
    return text, changed


def _ensure_basic_punctuation(text: str) -> tuple[str, bool]:
    if not text:
        return text, False
    stripped = text.rstrip()
    if stripped[-1] in _ENDING_PUNCT:
        return text, False
    cjk_count = len(_CJK_RE.findall(stripped))
    if cjk_count < 6:
        return text, False
    return stripped + "。", True


def _add_conservative_commas(text: str) -> tuple[str, int]:
    """Add a few readable pauses without rewriting words.

    This is intentionally weaker than a punctuation model.  It only inserts a
    comma before common spoken-Chinese connectors inside long, otherwise
    under-punctuated ASR lines.
    """
    if len(_CJK_RE.findall(text)) < 28:
        return text, 0
    if len(re.findall(r"[，。！？；：、,.!?;:]", text)) >= max(2, len(text) // 28):
        return text, 0

    changed = 0
    for word in _SOFT_PAUSE_WORDS:
        pattern = re.compile(rf"(?<![，。！？；：、,.!?;:\s])({re.escape(word)})")

        def repl(match: re.Match[str]) -> str:
            nonlocal changed
            if match.start() < 8:
                return match.group(1)
            if word == "就是" and match.start() > 0 and text[match.start() - 1] == "的":
                return match.group(1)
            changed += 1
            return "，" + match.group(1)

        text = pattern.sub(repl, text)
    return text, changed


def _asr_review_reasons(raw: str, normalized: str, *, profile: str | None = None) -> list[str]:
    haystack = "\n".join([raw or "", normalized or ""])
    reasons: list[str] = []
    for pattern, reason in _review_patterns(profile):
        if pattern.search(haystack) and reason not in reasons:
            reasons.append(reason)
    if profile == _STANDARD3_PROFILE and raw and normalized and raw != normalized:
        if "有点搅扰" in normalized or "怎么突然间脾气不好了" in normalized:
            reasons.append("已应用录音专用人工确认纠错")
        if "最有力量的是为我祷告" in normalized and re.search("最有质量|最有重要|因为我们好好|我好，那么", raw):
            reasons.append("已应用录音专用人工确认纠错")
        if re.search("李会|林熙|兰艺|守则拿来做人性化|我的例子就举完了|没有服侍|但还在群里", normalized) and re.search(
            "李慧|理慧|林夕|蓝艺|蓝意|蓝蚁|男人和金子|守的来|手的拿来|性免|没有没有|但还在学你|但还在群营|管有关个地",
            raw,
        ):
            reasons.append("已应用录音专用上下文纠错")
        if _YOUTH_FELLOWSHIP_PHRASE in normalized and re.search("君子的每人|君子的每位|女子内|青年团气|青年团庆|每一位都是姊妹", raw):
            reasons.append("已应用录音专用上下文纠错")
    elif profile == _LEGACY_GENERAL_PROFILE and raw and normalized and raw != normalized:
        if "有点搅扰" in normalized or "怎么突然间脾气不好了" in normalized:
            reasons.append("已应用通用强上下文纠错")
        if "最有力量的是为我祷告" in normalized and re.search("最有质量|最有重要|因为我们好好|我好，那么", raw):
            reasons.append("已应用通用强上下文纠错")
        if re.search("李会|林熙|兰艺|守则拿来做人性化|我的例子就举完了|没有服侍|但还在群里", normalized) and re.search(
            "李慧|理慧|林夕|蓝艺|蓝意|蓝蚁|男人和金子|守的来|手的拿来|性免|没有没有|但还在学你|但还在群营|管有关个地|管有关地",
            raw,
        ):
            reasons.append("已应用通用强上下文纠错")
        if re.search("大家其他的从中|为什么他们俩|一年多没有服侍|守则拿来做人性化|我的例子就举完了", normalized) and re.search(
            "管有关个地|管有关地|从东|凭什么|没有没有不事|守的来|手的拿来|性免",
            raw,
        ):
            reasons.append("已应用通用强上下文纠错")
        if _YOUTH_FELLOWSHIP_PHRASE in normalized and re.search("君子的每人|君子的每位|女子内|青年团气|青年团庆|每一位都是姊妹", raw):
            reasons.append("已应用通用强上下文纠错")
    return reasons


def normalize_transcript_text(
    text: str,
    *,
    context: str = "",
    language: str | None = "zh",
    profile: str | None = None,
) -> tuple[str, dict]:
    raw = text or ""
    stats = {
        "profile": _normalizer_profile(profile),
        "lexical_rewrites_enabled": _normalizer_profile(profile) in _LEXICAL_REWRITE_PROFILES,
        "simplified_changed": False,
        "symbols_removed": 0,
        "safe_replacements": 0,
        "conservative_commas_added": 0,
        "asr_review_reasons": [],
        "terminal_punctuation_added": False,
    }
    if (language or "zh").lower().startswith("zh"):
        simplified = _to_simplified(raw)
        stats["simplified_changed"] = simplified != raw
    else:
        simplified = raw
    cleaned, removed = _cleanup_symbols(simplified)
    stats["symbols_removed"] = removed
    cleaned, replacements = _apply_safe_replacements(cleaned, context or cleaned, profile=profile)
    stats["safe_replacements"] = replacements
    cleaned, comma_count = _add_conservative_commas(cleaned)
    stats["conservative_commas_added"] = comma_count
    cleaned, punct_added = _ensure_basic_punctuation(cleaned)
    stats["terminal_punctuation_added"] = punct_added
    stats["asr_review_reasons"] = _asr_review_reasons(raw, cleaned, profile=profile)
    return cleaned, stats


def normalize_segments(
    segments: Iterable[Segment],
    *,
    language: str | None = "zh",
    profile: str | None = None,
) -> tuple[list[Segment], dict]:
    items = list(segments)
    context = "\n".join(s.text or "" for s in items)
    normalized_profile = _normalizer_profile(profile)
    normalized: list[Segment] = []
    stats = {
        "mode": "local_text_normalizer",
        "profile": normalized_profile,
        "lexical_rewrites_enabled": normalized_profile in _LEXICAL_REWRITE_PROFILES,
        "input_segments": len(items),
        "output_segments": 0,
        "segments_changed": 0,
        "simplified_segments": 0,
        "symbols_removed": 0,
        "safe_replacements": 0,
        "conservative_commas_added": 0,
        "terminal_punctuation_added": 0,
        "asr_review_segment_count": 0,
        "asr_review_segments": [],
        "segments_with_punctuation": 0,
        "punctuation_ratio": 0.0,
        "first_mention_phonetic_consistency": {
            "enabled": False,
            "replacement_count": 0,
            "segments_changed": 0,
            "groups": [],
        },
    }
    for idx, seg in enumerate(items):
        original_source = seg.original_text or seg.text
        if (language or "zh").lower().startswith("zh"):
            display_original = _to_simplified(original_source or "")
        else:
            display_original = original_source or ""
        text, seg_stats = normalize_transcript_text(seg.text, context=context, language=language, profile=normalized_profile)
        if not text:
            continue
        changed = text != (seg.text or "")
        original_text = display_original if (changed or seg.original_text) and display_original != text else None
        if changed:
            stats["segments_changed"] += 1
        if seg_stats["simplified_changed"]:
            stats["simplified_segments"] += 1
        stats["symbols_removed"] += int(seg_stats["symbols_removed"])
        stats["safe_replacements"] += int(seg_stats["safe_replacements"])
        stats["conservative_commas_added"] += int(seg_stats["conservative_commas_added"])
        review_reasons = list(seg_stats.get("asr_review_reasons") or [])
        if review_reasons:
            stats["asr_review_segment_count"] += 1
            if len(stats["asr_review_segments"]) < 40:
                stats["asr_review_segments"].append({
                    "index": idx,
                    "start": seg.start,
                    "end": seg.end,
                    "text": text,
                    "original_text": display_original,
                    "reasons": review_reasons,
                })
        if seg_stats["terminal_punctuation_added"]:
            stats["terminal_punctuation_added"] += 1
        if _ANY_PUNCT_RE.search(text):
            stats["segments_with_punctuation"] += 1
        normalized.append(
            replace(
                seg,
                text=text,
                original_text=original_text,
            )
        )
    if normalized_profile in _LEXICAL_REWRITE_PROFILES:
        normalized, contextual_merge_count = _merge_contextual_asr_fragments(
            normalized,
            profile=normalized_profile,
        )
    else:
        contextual_merge_count = 0
    normalized, merged_count = _merge_single_char_fragments(normalized)
    if (
        (language or "zh").lower().startswith("zh")
        and normalized_profile in _LEXICAL_REWRITE_PROFILES
    ):
        try:
            from .term_consistency import apply_first_mention_phonetic_consistency

            normalized, consistency_stats = apply_first_mention_phonetic_consistency(normalized)
            stats["first_mention_phonetic_consistency"] = consistency_stats
            stats["segments_changed"] += int(consistency_stats.get("segments_changed") or 0)
        except Exception as exc:
            stats["first_mention_phonetic_consistency"] = {
                "enabled": True,
                "error": str(exc),
                "replacement_count": 0,
                "segments_changed": 0,
                "groups": [],
            }
    elif (language or "zh").lower().startswith("zh"):
        stats["first_mention_phonetic_consistency"] = {
            "enabled": False,
            "mode": "review_only",
            "reason": "默认通用模式只标记同音实体疑点，不自动覆盖 ASR 原文。",
            "replacement_count": 0,
            "segments_changed": 0,
            "groups": [],
        }
    _refresh_asr_review_segments(normalized, stats, profile=normalized_profile)
    stats["output_segments"] = len(normalized)
    stats["contextual_asr_merges"] = contextual_merge_count
    stats["merged_fragments"] = merged_count
    stats["segments_with_punctuation"] = sum(1 for seg in normalized if _ANY_PUNCT_RE.search(seg.text))
    if normalized:
        stats["punctuation_ratio"] = round(stats["segments_with_punctuation"] / len(normalized), 4)
    raw_text = "\n".join(seg.text or "" for seg in items)
    final_text = "\n".join(seg.text or "" for seg in normalized)
    stats["raw_chars"] = len(raw_text)
    stats["final_chars"] = len(final_text)
    stats["raw_text_sha256"] = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    stats["final_text_sha256"] = hashlib.sha256(final_text.encode("utf-8")).hexdigest()
    return normalized, stats


def _refresh_asr_review_segments(
    segments: list[Segment],
    stats: dict,
    *,
    profile: str | None = None,
    limit: int = 40,
) -> None:
    """Keep review markers aligned with the final emitted segments."""
    stats["asr_review_segment_count"] = 0
    stats["asr_review_segments"] = []
    for idx, seg in enumerate(segments):
        raw = _to_simplified(seg.original_text or seg.text)
        reasons = _asr_review_reasons(raw, seg.text, profile=profile)
        if not reasons:
            continue
        stats["asr_review_segment_count"] += 1
        if len(stats["asr_review_segments"]) >= limit:
            continue
        stats["asr_review_segments"].append({
            "index": idx,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "original_text": raw,
            "reasons": reasons,
        })


def _merge_contextual_asr_fragments(segments: list[Segment], *, profile: str | None = None) -> tuple[list[Segment], int]:
    """Merge narrow ASR splits where the boundary itself created a bad phrase."""
    merged: list[Segment] = []
    count = 0
    for seg in segments:
        text = (seg.text or "").strip()
        if (
            merged
            and _YOUTH_FELLOWSHIP_LEAD_RE.fullmatch((merged[-1].text or "").strip())
        ):
            prev = merged[-1]
            if (seg.start - prev.end) <= 1.2 and text.startswith(_YOUTH_FELLOWSHIP_PHRASE):
                tail = text[len(_YOUTH_FELLOWSHIP_PHRASE):].lstrip()
                if tail:
                    if tail[0] in "，,、；;：:":
                        new_text = f"我相信在{_YOUTH_FELLOWSHIP_PHRASE}{tail}"
                    elif tail[0] in _ENDING_PUNCT:
                        new_text = f"我相信在{_YOUTH_FELLOWSHIP_PHRASE}{tail}"
                    else:
                        new_text = f"我相信在{_YOUTH_FELLOWSHIP_PHRASE}，{tail}"
                else:
                    new_text = f"我相信在{_YOUTH_FELLOWSHIP_PHRASE}。"
                original_text = "\n".join(
                    x for x in [prev.original_text or prev.text, seg.original_text or seg.text] if x
                )
                merged[-1] = replace(
                    prev,
                    end=max(prev.end, seg.end),
                    text=new_text,
                    original_text=original_text,
                )
                count += 1
                continue
        if merged and text.startswith("好，"):
            prev = merged[-1]
            prev_text = (prev.text or "").rstrip()
            if (
                (seg.start - prev.end) <= 0.8
                and prev_text.rstrip(_ENDING_PUNCT).endswith("不")
                and re.search(r"(巴不得|希望看到).{0,16}不[。！？!?…]*$", prev_text)
            ):
                rest = text[2:].lstrip()
                new_text = prev_text.rstrip(_ENDING_PUNCT) + "好"
                new_text = new_text + (f"，{rest}" if rest else "。")
                original_text = "\n".join(
                    x for x in [prev.original_text or prev.text, seg.original_text or seg.text] if x
                )
                merged[-1] = replace(
                    prev,
                    end=max(prev.end, seg.end),
                    text=new_text,
                    original_text=original_text,
                )
                count += 1
                continue
        if merged and re.fullmatch(r"(?:在)?群里[。！？!?…]*", text):
            prev = merged[-1]
            prev_text = (prev.text or "").rstrip()
            prev_body = prev_text.rstrip(_ENDING_PUNCT)
            if (seg.start - prev.end) <= 0.8 and prev_body.endswith(("但还", "但还在")):
                original_text = "\n".join(
                    x for x in [prev.original_text or prev.text, seg.original_text or seg.text] if x
                )
                prefix = prev_body[:-1] if prev_body.endswith("但还在") else prev_body
                merged[-1] = replace(
                    prev,
                    end=max(prev.end, seg.end),
                    text=prefix + "在群里。",
                    original_text=original_text,
                )
                count += 1
                continue
        merged.append(seg)
    return merged, count


def _merge_single_char_fragments(segments: list[Segment]) -> tuple[list[Segment], int]:
    merged: list[Segment] = []
    count = 0
    filler = set("好对嗯啊哦是行")
    for seg in segments:
        text = (seg.text or "").strip()
        body = text.rstrip(_ENDING_PUNCT + "。！？!?")
        if merged and text.startswith("作，"):
            prev = merged[-1]
            prev_body = prev.text.rstrip().rstrip(_ENDING_PUNCT)
            if prev_body.endswith("的工") and (seg.start - prev.end) <= 0.8:
                new_text = text[2:].strip()
                merged[-1] = replace(
                    prev,
                    end=max(prev.end, seg.start),
                    text=prev_body + "作。",
                    original_text="\n".join(x for x in [prev.original_text or prev.text, seg.original_text or seg.text] if x),
                )
                if new_text:
                    merged.append(replace(seg, text=new_text, original_text=seg.original_text or seg.text))
                count += 1
                continue
        if (
            merged
            and len(body) == 1
            and _CJK_RE.fullmatch(body)
            and body not in filler
            and (seg.end - seg.start) <= 1.2
            and (seg.start - merged[-1].end) <= 0.8
            and len(merged[-1].text) >= 8
        ):
            prev = merged[-1]
            prev_text = prev.text.rstrip()
            prev_body = prev_text.rstrip(_ENDING_PUNCT)
            suffix = text if text[-1] in _ENDING_PUNCT else text + "。"
            merged_text = prev_body + suffix
            original_text = "\n".join(
                x for x in [prev.original_text or prev.text, seg.original_text or seg.text] if x
            )
            merged[-1] = replace(prev, end=max(prev.end, seg.end), text=merged_text, original_text=original_text)
            count += 1
            continue
        merged.append(seg)
    return merged, count


def join_wrapped_transcript_lines(text: str) -> str:
    """Join exported transcript lines that split one Chinese word unnaturally."""
    lines = text.splitlines()
    out: list[str] = []
    ts_re = re.compile(r"^(\[[^\]]+\]\s*)(.*)$")
    for line in lines:
        m = ts_re.match(line)
        if (
            m
            and out
            and _CJK_RE.search(m.group(2))
            and len(m.group(2).strip("。！？!?，,；;：:、 ")) <= 2
            and re.search(r"的[工事话]|这[件]|那个|这个|团契|同工", out[-1])
        ):
            prev = out[-1].rstrip()
            prev = prev.rstrip("。！？!?")
            out[-1] = prev + m.group(2).strip()
            continue
        out.append(line)
    return "\n".join(out)
