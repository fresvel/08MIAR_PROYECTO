from traitlets.config import get_config

# Use pdflatex instead of xelatex to avoid the missing texlive-xetex package.
c = get_config()
c.PDFExporter.latex_command = [
    "pdflatex",
    "{filename}",
    "-interaction=nonstopmode",
]
