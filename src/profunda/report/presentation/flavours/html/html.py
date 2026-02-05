from profunda.report.presentation.core import HTML


class HTMLHTML(HTML):
    def render(self) -> str:
        return self.content["html"]
