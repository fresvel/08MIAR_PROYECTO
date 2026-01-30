from traitlets.config import get_config

# Force nbconvert to use pdflatex instead of xelatex when exporting via JupyterLab.
c = get_config()
c.PDFExporter.latex_command = [
    "pdflatex",
    "{filename}",
    "-interaction=nonstopmode",
]
c.LatexExporter.latex_command = [
    "pdflatex",
    "{filename}",
    "-interaction=nonstopmode",
]
