# Static snapshot of Silero TTS model bundles, grouped by language key.
# Replaces reading latest_silero_models.yml at runtime (which needed OmegaConf).
# Values are the bundle ids accepted as `speaker=` / validated against --speaker.
SILERO_TTS_BUNDLES: dict[str, tuple[str, ...]] = {
    "ru": (
        "v5_turkic", "v5_caucasian", "v5_cis_base", "v5_cis_base_nostress_jit",
        "v5_cis_base_nostress", "v5_cis_ext", "v5_5_ru", "v5_4_ru", "v5_3_ru",
        "v5_2_ru", "v5_1_ru", "v5_ru", "v4_ru", "v3_1_ru", "ru_v3", "aidar_v2",
        "aidar_8khz", "aidar_16khz", "baya_v2", "baya_8khz", "baya_16khz",
        "irina_v2", "irina_8khz", "irina_16khz", "kseniya_v2", "kseniya_8khz",
        "kseniya_16khz", "natasha_v2", "natasha_8khz", "natasha_16khz",
        "ruslan_v2", "ruslan_8khz", "ruslan_16khz",
    ),
    "en": ("v3_en", "v3_en_indic", "lj_v2", "lj_8khz", "lj_16khz"),
    "de": ("v3_de", "thorsten_v2", "thorsten_8khz", "thorsten_16khz"),
    "es": ("v3_es", "tux_v2", "tux_8khz", "tux_16khz"),
    "fr": ("v3_fr", "gilles_v2", "gilles_8khz", "gilles_16khz"),
    "ba": ("aigul_v2",),
    "xal": ("v3_xal", "erdni_v2"),
    "tt": ("v3_tt", "dilyara_v2"),
    "uz": ("v4_uz", "v3_uz", "dilnavoz_v2"),
    "ua": ("v4_ua", "v3_ua", "mykyta_v2"),
    "indic": ("v4_indic", "v3_indic"),
    "cyrillic": ("v4_cyrillic",),
    "multi": ("multi_v2",),
}


def known_bundle_ids() -> frozenset[str]:
    """All Silero TTS bundle ids across every language."""
    return frozenset(b for bundles in SILERO_TTS_BUNDLES.values() for b in bundles)
