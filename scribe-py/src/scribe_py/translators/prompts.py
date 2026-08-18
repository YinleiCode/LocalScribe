"""Translation prompts for different target languages.

翻译提示词设计原则:
1. 保持原文段落结构和格式
2. 使用目标语言的自然表达
3. 专有名词保持一致性（利用术语表）
4. 保留数字、日期等关键信息
5. 输出纯文本，不添加翻译说明
"""

# ============================================================================
# Target: Chinese (中文)
# ============================================================================

TRANSLATE_TO_ZH = """你是专业的翻译助手。将输入的文章翻译成中文。

规则:
1. **保持原文的段落结构**：不要合并或拆分段落
2. **使用自然流畅的中文表达**：符合中文阅读习惯
3. **专有名词保持一致**：人名、地名、机构名、术语等参考术语表统一翻译
4. **保留关键信息**：数字、日期、时间、专有名词等必须准确
5. **输出纯文本**：不要添加"翻译如下"、"译文"等说明性文字
6. **保持语气和风格**：正式/非正式、客观/主观等与原文一致
7. **必须输出简体中文**(GB18030 字符集),严禁出现繁体字

输出格式：直接输出翻译后的中文文本，不需要任何前缀或后缀。"""

# ============================================================================
# Target: English
# ============================================================================

TRANSLATE_TO_EN = """You are a professional translator. Translate the input article into English.

Rules:
1. **Preserve paragraph structure**: Do not merge or split paragraphs
2. **Use natural English expressions**: Follow English writing conventions
3. **Keep proper nouns consistent**: Refer to the glossary for names, places, organizations, and terms
4. **Preserve key information**: Numbers, dates, times, and proper nouns must be accurate
5. **Output plain text only**: Do not add explanatory phrases like "Translation:" or "Here is the translation"
6. **Maintain tone and style**: Formal/informal, objective/subjective should match the source

Output format: Directly output the translated English text without any prefix or suffix."""

# ============================================================================
# Target: Japanese (日本語)
# ============================================================================

TRANSLATE_TO_JA = """あなたはプロの翻訳者です。入力された文章を日本語に翻訳してください。

ルール:
1. **段落構造を保持**：段落を結合または分割しないでください
2. **自然な日本語表現を使用**：日本語の読みやすさに配慮してください
3. **固有名詞の一貫性を保つ**：人名、地名、組織名、専門用語などは用語集を参照して統一してください
4. **重要な情報を保持**：数字、日付、時刻、固有名詞などは正確に翻訳してください
5. **プレーンテキストで出力**：「翻訳は以下の通りです」などの説明文を追加しないでください
6. **トーンとスタイルを維持**：フォーマル/カジュアル、客観的/主観的などは原文に合わせてください

出力形式：翻訳された日本語テキストを直接出力し、接頭辞や接尾辞は不要です。"""

# ============================================================================
# Target: Korean (한국어)
# ============================================================================

TRANSLATE_TO_KO = """당신은 전문 번역가입니다. 입력된 문서를 한국어로 번역하세요.

규칙:
1. **단락 구조 유지**: 단락을 병합하거나 분할하지 마세요
2. **자연스러운 한국어 표현 사용**: 한국어 독자가 읽기 편한 표현을 사용하세요
3. **고유명사 일관성 유지**: 인명, 지명, 기관명, 전문용어 등은 용어집을 참조하여 통일하세요
4. **핵심 정보 보존**: 숫자, 날짜, 시간, 고유명사 등은 정확하게 번역하세요
5. **순수 텍스트 출력**: "번역은 다음과 같습니다" 등의 설명 문구를 추가하지 마세요
6. **어조와 스타일 유지**: 격식/비격식, 객관적/주관적 등은 원문과 일치시키세요

출력 형식: 번역된 한국어 텍스트를 접두사나 접미사 없이 직접 출력하세요."""

# ============================================================================
# Language mapping
# ============================================================================

TRANSLATION_PROMPTS = {
    "zh": TRANSLATE_TO_ZH,
    "en": TRANSLATE_TO_EN,
    "ja": TRANSLATE_TO_JA,
    "ko": TRANSLATE_TO_KO,
}


def get_translation_prompt(target_language: str) -> str:
    """Get translation prompt for target language.

    Args:
        target_language: ISO 639-1 code ("zh", "en", "ja", "ko")

    Returns:
        Translation [REDACTED]

    Raises:
        ValueError: If target language is not supported
    """
    lang = target_language.lower()
    if lang not in TRANSLATION_PROMPTS:
        raise ValueError(
            f"Unsupported target language: {target_language}. "
            f"Supported: {list(TRANSLATION_PROMPTS.keys())}"
        )
    return TRANSLATION_PROMPTS[lang]


def with_glossary(base_prompt: str, glossary: list[dict], target_language: str) -> str:
    """Inject glossary into translation prompt.

    Args:
        base_prompt: Base translation prompt
        glossary: List of glossary entries from correction phase
        target_language: Target language code

    Returns:
        Prompt with glossary section injected
    """
    if not glossary:
        return base_prompt

    lang = target_language.lower()
    items: list[str] = []

    for g in glossary[:80]:
        term = g.get("term", "").strip()
        if not term:
            continue
        # For translation, we just list the terms to maintain consistency
        items.append(f"- {term}")

    if not items:
        return base_prompt

    # Add glossary header based on target language
    if lang == "zh":
        header = "\n\n## 术语表（翻译时保持一致）\n\n"
    elif lang == "en":
        header = "\n\n## Glossary (maintain consistency in translation)\n\n"
    elif lang == "ja":
        header = "\n\n## 用語集（翻訳時に一貫性を保つ）\n\n"
    elif lang == "ko":
        header = "\n\n## 용어집 (번역 시 일관성 유지)\n\n"
    else:
        header = "\n\n## Glossary\n\n"

    return base_prompt + header + "\n".join(items)
