from profunda.report.presentation.core.alerts import Alerts
from profunda.report.presentation.flavours.html import templates
from profunda.utils.styles import get_alert_styles


class HTMLAlerts(Alerts):
    def render(self) -> str:
        styles = get_alert_styles()

        return templates.template("alerts.html").render(**self.content, styles=styles)
