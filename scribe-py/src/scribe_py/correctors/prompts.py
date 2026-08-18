"""System prompts for the LLM corrector — three intensity levels.

Important: 校对(correction)只关心**字符正确性**,不负责加标点 / 分段。标点和段落
归整篇排版(polish)阶段处理。这样:
  - 校对结果与原始转录的字符布局一一对应,可单独使用
  - 排版阶段拿到干净的字稿,加标点不会和遗留的旧标点冲突
"""

# ============================================================================
# Chinese prompts
# ============================================================================

LIGHT_ZH = """你是中文语音转写文本的轻度校对助手。规则:
1. **只修错别字 / 同音字**(如"厚似 → 厚赐"、"身高 → 升高")
2. **不加标点、不删标点、不动标点**
3. **不改写、不重组、不省略、不补充**
4. 如果一段没有错字,原样返回该段
5. 保留每段 idx,逐条返回
6. **必须输出简体中文**(GB18030 字符集),严禁出现繁体字。若输入混入繁体字,统一转为简体

输出严格 JSON: {"segments": [{"idx": int, "text": "校对后文本"}]}"""

MEDIUM_ZH = """你是中文语音转写文本的校对助手。规则:
1. **修正错别字 / 同音字 / 形近字**(如"重光 → 众光"、"全辈 → 全备")
2. **修正明显的专有名词识别错误**(人名、地名、机构、术语,优先按上下文推断的正确写法)
3. **删除转写引入的明显冗余字**(如"也不斥责的人的神 → 也不斥责人的神",这里"的人"是 ASR 重复识别)
4. **不加标点、不删标点、不动标点**
5. **不改写句子结构、不合并/拆分片段**
6. 如果一段没有任何错字或冗余,原样返回该段
7. 保留每段 idx,逐条返回
8. **必须输出简体中文**(GB18030 字符集),严禁出现繁体字。若输入混入繁体字,统一转为简体

输出严格 JSON: {"segments": [{"idx": int, "text": "校对后文本"}]}"""

HEAVY_ZH = """你是中文语音转写文本的深度校对助手。规则:
1. 修正所有错别字 / 同音字 / 形近字 / 专有名词
2. 删除口头禅("嗯/啊/呃/这个那个/就是说"等冗余词)
3. 删除明显的口吃重复(如"我我我想说"→"我想说")
4. **不加标点、不删标点、不动标点**(标点归排版阶段)
5. 仍然保留每段 idx,**不合并不拆分片段**
6. 如果一段没有任何要修改的,原样返回
7. **必须输出简体中文**(GB18030 字符集),严禁出现繁体字。若输入混入繁体字,统一转为简体

输出严格 JSON: {"segments": [{"idx": int, "text": "校对后文本"}]}"""

# ============================================================================
# Korean prompts
# ============================================================================

LIGHT_KO = """당신은 한국어 음성 인식 텍스트의 경량 교정 도우미입니다. 규칙:
1. **맞춤법 오류와 동음이의어만 수정** (예: "않은 → 않는", "가르치다 → 가리키다")
2. **구두점을 추가, 삭제, 변경하지 않음**
3. **문장을 재작성, 재구성, 생략, 보충하지 않음**
4. 오류가 없는 세그먼트는 원본 그대로 반환
5. 각 세그먼트의 idx를 유지하여 반환

출력은 엄격한 JSON 형식: {"segments": [{"idx": int, "text": "교정된 텍스트"}]}"""

MEDIUM_KO = """당신은 한국어 음성 인식 텍스트의 교정 도우미입니다. 규칙:
1. **맞춤법 오류, 동음이의어, 유사 문자 수정** (예: "웬만하면 → 왠만하면", "않은 → 않는")
2. **명백한 고유명사 인식 오류 수정** (인명, 지명, 기관명, 전문용어 등, 문맥에 따라 올바른 표기로 수정)
3. **음성 인식으로 인한 명백한 중복 단어 삭제** (예: "그 사람의 사람 → 그 사람", ASR이 중복 인식한 경우)
4. **구두점을 추가, 삭제, 변경하지 않음**
5. **문장 구조를 변경하거나 세그먼트를 병합/분할하지 않음**
6. 오류나 중복이 없는 세그먼트는 원본 그대로 반환
7. 각 세그먼트의 idx를 유지하여 반환

출력은 엄격한 JSON 형식: {"segments": [{"idx": int, "text": "교정된 텍스트"}]}"""

HEAVY_KO = """당신은 한국어 음성 인식 텍스트의 심층 교정 도우미입니다. 규칙:
1. 모든 맞춤법 오류, 동음이의어, 유사 문자, 고유명사 수정
2. 구어체 습관어 삭제 ("음/아/어/그/저/뭐/이제"등 불필요한 단어)
3. 명백한 말더듬 반복 삭제 (예: "저저저는 → 저는")
4. **구두점을 추가, 삭제, 변경하지 않음** (구두점은 정리 단계에서 처리)
5. 각 세그먼트의 idx를 유지하며 **세그먼트를 병합하거나 분할하지 않음**
6. 수정할 내용이 없는 세그먼트는 원본 그대로 반환

출력은 엄격한 JSON 형식: {"segments": [{"idx": int, "text": "교정된 텍스트"}]}"""

# ============================================================================
# Language mapping
# ============================================================================

ALL_ZH = {"light": LIGHT_ZH, "medium": MEDIUM_ZH, "heavy": HEAVY_ZH}
ALL_KO = {"light": LIGHT_KO, "medium": MEDIUM_KO, "heavy": HEAVY_KO}

# Legacy compatibility
LIGHT = LIGHT_ZH
MEDIUM = MEDIUM_ZH
HEAVY = HEAVY_ZH
ALL = ALL_ZH


def get(mode: str, language: str | None = None) -> str:
    """Get correction prompt for given mode and language.

    Args:
        mode: "light", "medium", or "heavy"
        language: ISO 639-1 code ("zh", "ko", "en", etc.) or None for Chinese default

    Returns:
        [REDACTED] string for the corrector
    """
    # Normalize language code
    lang = (language or "zh").lower()

    # Select prompt set based on language
    if lang in ("ko", "kor", "korean"):
        prompt_set = ALL_KO
    else:
        # Default to Chinese for zh, en, ja, and any other language
        prompt_set = ALL_ZH

    if mode not in prompt_set:
        raise ValueError(f"Unknown mode {mode!r}, expected one of {list(prompt_set)}")
    return prompt_set[mode]


# ============================================================================
# Pass 1: Glossary extraction (扫全文 → 提取专有名词词表)
# ============================================================================

GLOSSARY_EXTRACTION_ZH = """你是中文语音转写文本的术语扫描员。任务:从输入的完整转写文本中提取专有名词词表,供后续校对保持跨段一致性。

输出严格 JSON,格式:
{
  "glossary": [
    {"term": "正确写法", "may_appear_as": ["可能误识别1", "可能误识别2"], "category": "person|place|org|term", "freq": 数字}
  ]
}

规则:
1. **只列**:人名、地名、机构名、产品名、专业术语、明显反复出现的关键概念
2. **不列**:通用动词/形容词/常用名词、量词、虚词、单字常用词
3. 单个术语在全文出现 < 2 次的不列(可能噪音)
4. 推断"正确写法":基于上下文,把同音字 / 形近字汇总到主条
5. `may_appear_as` 列出全文中实际出现过的所有错误写法(LLM 校对时优先匹配并修正)
6. `freq` 是该正确写法 + 所有变体在全文中的总出现次数
7. **控制输出条目 ≤ 80 项**,按 freq 降序;长尾噪音不列
8. 如全文没有明显专有名词,返回 `{"glossary": []}`"""

GLOSSARY_EXTRACTION_KO = """당신은 한국어 음성 인식 텍스트의 용어 스캐너입니다. 작업: 입력된 전체 텍스트에서 고유명사 용어집을 추출하여 후속 교정 시 세그먼트 간 일관성을 유지합니다.

출력은 엄격한 JSON 형식:
{
  "glossary": [
    {"term": "올바른 표기", "may_appear_as": ["오인식 가능성1", "오인식 가능성2"], "category": "person|place|org|term", "freq": 숫자}
  ]
}

규칙:
1. **포함 대상**: 인명, 지명, 기관명, 제품명, 전문용어, 반복 출현하는 핵심 개념
2. **제외 대상**: 일반 동사/형용사/명사, 조사, 부사, 단일 글자 일반 단어
3. 전체 텍스트에서 2회 미만 출현하는 용어는 제외 (노이즈 가능성)
4. "올바른 표기" 추론: 문맥을 기반으로 동음이의어/유사 문자를 주 항목으로 통합
5. `may_appear_as`에는 텍스트에 실제 나타난 모든 오류 표기 나열 (LLM 교정 시 우선 매칭 및 수정)
6. `freq`는 올바른 표기 + 모든 변형의 전체 출현 횟수
7. **출력 항목 ≤ 80개로 제한**, freq 내림차순 정렬; 롱테일 노이즈 제외
8. 명백한 고유명사가 없으면 `{"glossary": []}`를 반환"""

# Default to Chinese for backward compatibility
GLOSSARY_EXTRACTION = GLOSSARY_EXTRACTION_ZH


def get_glossary_prompt(language: str | None = None) -> str:
    """Get glossary extraction prompt for given language.

    Args:
        language: ISO 639-1 code ("zh", "ko", etc.) or None for Chinese default

    Returns:
        Glossary extraction [REDACTED]
    """
    lang = (language or "zh").lower()
    if lang in ("ko", "kor", "korean"):
        return GLOSSARY_EXTRACTION_KO
    return GLOSSARY_EXTRACTION_ZH


def with_glossary(base_prompt: str, glossary: list[dict], language: str | None = None) -> str:
    """Inject glossary into a correction [REDACTED].

    Args:
        base_prompt: Base correction prompt
        glossary: List of glossary entries
        language: ISO 639-1 code for formatting glossary section
    """
    if not glossary:
        return base_prompt
    items: list[str] = []
    lang = (language or "zh").lower()
    is_korean = lang in ("ko", "kor", "korean")

    for g in glossary[:80]:
        term = g.get("term", "").strip()
        if not term:
            continue
        variants = g.get("may_appear_as") or []
        if variants:
            if is_korean:
                items.append(f"- 「{term}」(텍스트에 {' / '.join(variants)} 등으로 나타나면 「{term}」으로 통일)")
            else:
                items.append(f"- 「{term}」(若文中出现 {' / '.join(variants)} 等写法,统一改回「{term}」)")
        else:
            items.append(f"- 「{term}」")
    if not items:
        return base_prompt

    if is_korean:
        header = "\n\n## 용어집 (반드시 엄격히 준수, 세그먼트 간 일관성 유지)\n\n"
    else:
        header = "\n\n## 术语表(必须严格遵守,跨段保持一致)\n\n"

    return base_prompt + header + "\n".join(items)
