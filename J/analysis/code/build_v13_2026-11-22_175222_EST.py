"""Build manuscript v13 (Word and PDF) from its Markdown source, following the v12 builder.

Usage: python build_v13_<stamp>.py <v13 bundle dir> <destination dir> <source_v4 dir (reference.docx, latex_header.tex)>
       <v5 code dir (word_layout.py)> [--pdf-engine xelatex] [--mainfont "Liberation Serif"] [--sansfont "Liberation Sans"]
On a Mac with Times New Roman and tectonic, pass --pdf-engine tectonic --mainfont "Times New Roman" --sansfont Arial.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree as E

ST = "2026-11-22_175222_EST"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("bundle", type=Path)
p.add_argument("destination", type=Path)
p.add_argument("source_v4", type=Path)
p.add_argument("v5_code", type=Path)
p.add_argument("--pdf-engine", default="xelatex")
p.add_argument("--mainfont", default="Liberation Serif")
p.add_argument("--sansfont", default="Liberation Sans")
a = p.parse_args()
bundle, dest = a.bundle.resolve(), a.destination.resolve()
dest.mkdir(parents=True, exist_ok=True)
source = bundle / f"manuscript_v13_{ST}.md"
sys.path.insert(0, str(a.v5_code.resolve()))
from word_layout import compact_wide_tables  # noqa: E402

base = [ "pandoc", str(source), "--resource-path", str(bundle)]

# Word
from docx import Document  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402

docx_path = dest / f"J_manuscript_bioRxiv_v13_{ST}.docx"
with tempfile.TemporaryDirectory() as tmp:
    reference = Document(str(a.source_v4 / "reference.docx"))
    for section in reference.sections:
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        page = OxmlElement("w:fldSimple")
        page.set(qn("w:instr"), "PAGE")
        footer._p.append(page)
    ref = Path(tmp) / "reference.docx"
    reference.save(ref)
    subprocess.run(base + ["--reference-doc", str(ref), "-o", str(docx_path)], check=True)
compact_wide_tables(docx_path)
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
q = lambda n: "{" + ns["w"] + "}" + n  # noqa: E731
tmp_path = dest / (docx_path.name + ".tmp")
with ZipFile(docx_path) as src, ZipFile(tmp_path, "w", compression=ZIP_DEFLATED) as out:
    for item in src.infolist():
        data = src.read(item)
        if item.filename == "word/document.xml":
            xml = E.fromstring(data)
            for sect in xml.iter(q("sectPr")):  # continuous line numbers, as in the PDF
                ln = sect.find(q("lnNumType"))
                if ln is None:
                    ln = E.Element(q("lnNumType"))
                    pg = sect.find(q("pgMar"))
                    (pg.addnext(ln) if pg is not None else sect.append(ln))
                ln.set(q("countBy"), "1"); ln.set(q("restart"), "continuous"); ln.set(q("distance"), "360")
            for para in xml.xpath("//w:p[.//w:drawing]", namespaces=ns):  # keep each picture with its caption
                props = para.find(q("pPr"))
                if props is None:
                    props = E.Element(q("pPr"))
                    para.insert(0, props)
                if props.find(q("keepNext")) is None:
                    E.SubElement(props, q("keepNext"))
                caption = para.getnext()
                if caption is not None and caption.tag == q("p"):
                    style = caption.find("./" + q("pPr") + "/" + q("pStyle"))
                    if style is not None and style.get(q("val")) == "ImageCaption":
                        cap_props = caption.find(q("pPr"))
                        if cap_props.find(q("keepLines")) is None:
                            E.SubElement(cap_props, q("keepLines"))
            data = E.tostring(xml, xml_declaration=True, encoding="UTF-8", standalone=True)
        out.writestr(item, data)
tmp_path.replace(docx_path)
print("wrote", docx_path)

# PDF
pdf_path = dest / f"J_manuscript_bioRxiv_v13_{ST}.pdf"
subprocess.run(base + ["--syntax-highlighting=none" if False else "--no-highlight", f"--pdf-engine={a.pdf_engine}",
                       "-V", "geometry:margin=1in", "-V", "fontsize=11pt", "-V", f"mainfont={a.mainfont}",
                       "-V", f"sansfont={a.sansfont}", "-V", "linestretch=1.15", "-V", "colorlinks=true",
                       "-V", "linkcolor=black", "-V", "urlcolor=blue", "-H", str(Path(__file__).resolve().parent / "latex_header_v13.tex"),
                       "-o", str(pdf_path)], check=True)
print("wrote", pdf_path)
