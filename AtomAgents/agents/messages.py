from langchain_core.messages import AIMessage


class ReportMessage(AIMessage):
    """AIMessage subclass marking an agent's final hand-off report.

    Keeps ``type == "ai"`` so provider role-mapping is unaffected, and stamps a
    marker into ``additional_kwargs`` so the subclass survives serialization to
    the memory DB and back (a bare subclass silently downgraded to AIMessage on
    every round-trip).
    """

    def __init__(self, content=None, **kwargs):
        additional_kwargs = dict(kwargs.pop("additional_kwargs", None) or {})
        additional_kwargs.setdefault("atom_message_type", "report")
        super().__init__(content=content, additional_kwargs=additional_kwargs, **kwargs)


__all__ = ["ReportMessage"]
