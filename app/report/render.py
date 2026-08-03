"""HTML->PDF rendering. WeasyPrint is the intended production renderer
(clean CSS-driven layout, per the PRD tech stack); it needs native
GTK/Pango/cairo libraries that this Windows dev box doesn't have, so
xhtml2pdf (pure Python, no native deps) is the fallback used here instead.
Both take the same rendered HTML string, so nothing else in the pipeline
needs to know which one actually ran.
"""

import logging
from io import BytesIO
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from app.report.build import ReportContext

logger = logging.getLogger("vie.report")

_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")))


def render_html(ctx: ReportContext) -> str:
    template = _env.get_template("report.html.jinja2")
    return template.render(ctx=ctx)


def render_pdf(html: str) -> bytes:
    try:
        from weasyprint import HTML

        return HTML(string=html).write_pdf()
    except Exception as exc:  # noqa: BLE001 - native-lib import failures land here on Windows dev boxes
        logger.warning("WeasyPrint unavailable (%s), falling back to xhtml2pdf", exc)
        from xhtml2pdf import pisa

        buffer = BytesIO()
        pisa.CreatePDF(html, dest=buffer)
        return buffer.getvalue()
