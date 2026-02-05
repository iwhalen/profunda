from profunda.report.presentation.core import VariableInfo
from profunda.report.presentation.flavours.html import templates


class HTMLVariableInfo(VariableInfo):
    def render(self) -> str:
        return templates.template("variable_info.html").render(**self.content)
