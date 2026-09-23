from enum import Enum
from typing import Literal


# pylint: disable=C0103
class PLang(Enum):
    # Python name -> User input name
    Cpp = "C++"
    Csharp = "C#"
    Python = "Python"
    Java = "Java"
    Rust = "Rust"
    Go = "Go"
    Typescript = "Typescript"
    Javascript = "Javascript"
    Ruby = "Ruby"
    Php = "Php"
    Scala = "Scala"
    Kotlin = "Kotlin"

    def __str__(self):
        return str(self.value.lower())

    def __eq__(self, other: str):
        if isinstance(other, PLang):
            return self.value.lower() == other.value.lower()
        else:
            match = PLang.fuzzy_match(other)
            return self.value.lower() == match.lower()

    @staticmethod
    def get_all_plangs() -> list[str]:
        """Returns all avaliable programming languages

        Returns:
            ["c++", "c#", "python", ...]
        """
        return [p.value.lower() for p in PLang]

    @staticmethod
    def get_single_line_comment(plang: str) -> Literal["#", "//"]:
        if plang.lower() in ("python", "ruby"):
            return "#"
        else:
            return "//"

    @staticmethod
    def get_markdown_lang_name(plang: str) -> str:
        """Get proper language name for markdown code blocks"""
        mapping = {
            "c++": "cpp",
            "c#": "csharp",
            "javascript": "javascript",
            "typescript": "typescript",
            "python": "python",
            "java": "java",
            "rust": "rust",
            "go": "go",
            "ruby": "ruby",
            "php": "php",
            "scala": "scala",
            "kotlin": "kotlin",
        }
        return mapping.get(plang.lower())

    @staticmethod
    def get_display_lang_name(plang: str) -> str:
        """Get proper language name for display in text"""
        mapping = {
            "c++": "C++",
            "c#": "C#",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "python": "Python",
            "java": "Java",
            "rust": "Rust",
            "go": "Go",
            "ruby": "Ruby",
            "php": "php",
            "scala": "Scala",
            "kotlin": "Kotlin",
        }
        return mapping.get(plang.lower())

    @staticmethod
    def fuzzy_match(plang: str) -> str | None:

        plang = plang.strip().lower()
        mapping = {
            "c++": PLang.Cpp,
            "cpp": PLang.Cpp,
            "c#": PLang.Csharp,
            "csharp": PLang.Csharp,
            "javascript": PLang.Javascript,
            "js": PLang.Javascript,
            "typescript": PLang.Typescript,
            "ts": PLang.Typescript,
            "python": PLang.Python,
            "java": PLang.Java,
            "rust": PLang.Rust,
            "go": PLang.Go,
            "ruby": PLang.Ruby,
            "php": PLang.Php,
            "scala": PLang.Scala,
            "kotlin": PLang.Kotlin,
        }
        if plang not in mapping:
            raise ValueError(f"{plang} not found")

        match = mapping[plang]
        return match.value.lower()


PLANGS = PLang.get_all_plangs()
