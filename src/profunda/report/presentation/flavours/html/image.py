from profunda.report.presentation.core import Image
from profunda.report.presentation.flavours.html import templates


class HTMLImage(Image):
    def render(self) -> str:
        return templates.template("diagram.html").render(**self.content)
