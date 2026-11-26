"""Sovereignty-preserving LLM Calibrator vs. PID/EMA simulation benchmark."""

EXT_ENUMS: tuple[str, ...] = ("doc_slide", "doc_hwpx", "doc_md", "media_img")
CATEGORIES: tuple[str, ...] = ("cat_0", "cat_1", "cat_2", "cat_3")
OUTCOMES: tuple[str, ...] = ("TP", "FP", "FN")
FEATURE_DIM: int = 32
