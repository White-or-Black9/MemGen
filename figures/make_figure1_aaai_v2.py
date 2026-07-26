"""Create a non-destructive labeled-flow revision of the MemGen SVG."""
from pathlib import Path
import re
import cairosvg

base = Path(__file__).parent
source = (base / "figure1_architecture.svg").read_text(encoding="utf-8")

# Remove frozen captions, without altering the module geometry.
source = re.sub(r'<text x="[^"]+" y="[^"]+" text-anchor="middle" class="small">\(Frozen\)</text>', '', source)

# SKIP is a dashed conditional-return path; INVOKE is a solid execution signal.
source = source.replace('d="M264 135 L146 135 L146 278" class="control"', 'd="M264 135 L146 135 L146 278" class="control-dashed"')
source = source.replace('d="M950 135 L832 135 L832 278" class="control"', 'd="M950 135 L832 135 L832 278" class="control-dashed"')
source = source.replace('d="M1100 135 L1180 135 L1180 330 L1065 330 L1065 352" class="control-dashed"', 'd="M1100 135 L1180 135 L1180 330 L1065 330 L1065 352" class="control"')

# White rounded label backplates isolate labels from paths at final paper scale.
plates = {
    'SKIP': [(121, 136, 50, 24), (807, 136, 50, 24)],
    'INVOKE': [(473, 136, 78, 24), (1083, 136, 94, 24)],
    'Query with': [(1060, 239, 150, 27)],
    'Retrieve top-k': [(1040, 329, 165, 27)],
}
for key, boxes in plates.items():
    search_from = 0
    for x, y, w, h in boxes:
        idx = source.find(f'>{key}', search_from)
        if idx < 0:
            continue
        start = source.rfind('<text', 0, idx)
        source = source[:start] + f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" opacity="0.96" stroke="none"/>\n  ' + source[start:]
        search_from = idx + 1

out = base / "figure1_architecture_aaai_v2.svg"
preview = base / "figure1_architecture_aaai_v2_preview.png"
out.write_text(source, encoding="utf-8")
cairosvg.svg2png(bytestring=source.encode(), write_to=str(preview), output_width=2800, output_height=1280)
print(out)
print(preview)
