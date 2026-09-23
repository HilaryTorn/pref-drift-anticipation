"""Canonical project-to-Multi-LCB language naming.

Project identifiers are stable metadata keys. Multi-LCB identifiers control its
prompt renderer and output filenames, while Markdown fence identifiers control
the assistant target stored in SFT JSONL. C# is the one current language where
all three names are not identical, so callers must use these helpers rather
than interpolating a project identifier into an upstream path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LCBLanguage:
    project_id: str
    upstream_id: str
    markdown_fence: str


LANGUAGES = {
    "python": LCBLanguage("python", "python", "python"),
    "go": LCBLanguage("go", "go", "go"),
    "rust": LCBLanguage("rust", "rust", "rust"),
    "php": LCBLanguage("php", "php", "php"),
    "ruby": LCBLanguage("ruby", "ruby", "ruby"),
    "scala": LCBLanguage("scala", "scala", "scala"),
    "java": LCBLanguage("java", "java", "java"),
    "cpp": LCBLanguage("cpp", "c++", "cpp"),
    "csharp": LCBLanguage("csharp", "c#", "csharp"),
    "javascript": LCBLanguage("javascript", "javascript", "javascript"),
    "typescript": LCBLanguage("typescript", "typescript", "typescript"),
    "kotlin": LCBLanguage("kotlin", "kotlin", "kotlin"),
}


def language_spec(project_id: str) -> LCBLanguage:
    try:
        return LANGUAGES[project_id]
    except KeyError as exc:
        supported = ", ".join(sorted(LANGUAGES))
        raise ValueError(f"Unsupported project language {project_id!r}; expected one of: {supported}") from exc


def upstream_language(project_id: str) -> str:
    return language_spec(project_id).upstream_id


def markdown_fence(project_id: str) -> str:
    return language_spec(project_id).markdown_fence
