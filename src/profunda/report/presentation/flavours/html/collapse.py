from profunda.report.presentation.core import Collapse
from profunda.report.presentation.flavours.html import templates


class HTMLCollapse(Collapse):
    def render(self) -> str:
        return templates.template("collapse.html").render(**self.content)
