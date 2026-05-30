from storyforge.rag.generative_ai import Gen_mode, StoryType, get_mode_sampling, parse_gen_mode, parse_story_type

__all__ = [
    "Gen_mode",
    "StoryType",
    "parse_gen_mode",
    "parse_story_type",
    "get_mode_sampling",
    "RAG3StepResult",
    "generate_story_3step_langchain",
]

# langchain_rag needs LangChain + transformers. Load it only when you call
# generate_story_3step_langchain (PEP 562 lazy import).
_LAZY = {"RAG3StepResult", "generate_story_3step_langchain"}


def __getattr__(name):
    if name in _LAZY:
        from storyforge.rag import langchain_rag

        return getattr(langchain_rag, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
