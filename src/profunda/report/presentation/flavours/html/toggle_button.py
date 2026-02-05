from profunda.report.presentation.core import ToggleButton
from profunda.report.presentation.flavours.html import templates


class HTMLToggleButton(ToggleButton):
    def render(self) -> str:
        return templates.template("toggle_button.html").render(**self.content)
