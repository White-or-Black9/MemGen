from pathlib import Path
import re
import cairosvg

root = Path(__file__).parent
src = (root / 'figures/figure1_architecture.svg').read_text(encoding='utf-8')

def cell(x, y): return f'<rect x="{x}" y="{y}" width="10" height="17" rx="1" class="token-cell"/>'
def row(y, label):
    return f'<text x="1022" y="{y+13}" class="slot-label">{label}</text>' + ''.join(cell(1070+i*13, y) for i in range(8))

right = f'''  <!-- Right panel: Bank-left / Weaver-right layout -->
  <g id="right-panel">
    <text x="1055" y="39" text-anchor="middle" class="title-right">(b) Ours</text>
    <text x="1055" y="64" text-anchor="middle" class="title-right">Session-local latent memory management</text>
    <rect id="right-reasoner" x="744" y="278" width="164" height="78" rx="9" class="module"/>
    <text x="826" y="324" text-anchor="middle" class="label" font-weight="600">Reasoner</text>
    <text x="826" y="346" text-anchor="middle" class="small gray">(Frozen)</text>
    <rect id="right-trigger" x="950" y="98" width="150" height="74" rx="9" class="module"/>
    <text x="1025" y="143" text-anchor="middle" class="label" font-weight="600">Trigger</text>
    <text x="1025" y="161" text-anchor="middle" class="small gray">(Frozen)</text>

    <!-- One H_t trunk branches to trigger, bank, and Weaver -->
    <path d="M908 317 H930" class="trunk"/><circle cx="930" cy="317" r="3" fill="#343434"/>
    <path d="M930 317 V206 H1025 V172" class="control"/>
    <path d="M950 135 H832 V278" class="control-dashed"/>
    <rect x="806" y="136" width="52" height="24" rx="6" fill="#fff"/><text x="832" y="153" class="small gray" text-anchor="middle">SKIP</text>
    <path d="M1100 135 H1190 V274 H1290 V292" class="control-dashed"/>
    <rect x="1090" y="136" width="82" height="24" rx="6" fill="#fff"/><text x="1131" y="153" class="small gray" text-anchor="middle">INVOKE</text>

    <!-- Compact bank in the middle-left -->
    <rect id="memory-bank" x="1000" y="250" width="180" height="185" rx="11" class="bank"/>
    <text x="1090" y="274" class="bank-title" text-anchor="middle">Session-Local</text>
    <text x="1090" y="293" class="bank-title" text-anchor="middle">Latent Memory Bank</text>
    {row(305,'Slot 1')}{row(332,'Slot 2')}
    <text x="1045" y="367" class="ellipsis" text-anchor="middle">...</text>{row(375,'Slot N')}
    <line x1="1008" y1="405" x2="1172" y2="405" class="separator"/>
    <text x="1090" y="424" class="bank-foot" text-anchor="middle">Capacity-aware replacement</text>
    <rect id="reset-session" x="1028" y="195" width="124" height="32" rx="6" class="reset-box"/>
    <text x="1090" y="216" class="reset-text" text-anchor="middle">Reset at session end</text>
    <path d="M1090 250 V227" class="memory-dashed"/>

    <!-- Query H_t to Bank; retrieve R_t horizontally to Weaver -->
    <path d="M930 317 V240 H1055 V250" class="memory" marker-end="url(#arrow-purple)"/>
    <rect x="895" y="213" width="126" height="25" rx="6" fill="#fff"/><text x="958" y="230" class="memory-label" text-anchor="middle">Query with <tspan font-style="italic">H</tspan><tspan baseline-shift="sub" font-size="9">t</tspan></text>
    <path d="M1180 405 H1200 V358 H1215" class="memory" marker-end="url(#arrow-purple)"/>
    <rect x="1184" y="378" width="132" height="22" rx="6" fill="#fff"/><text x="1250" y="394" class="memory-label" text-anchor="middle">Retrieve top-k <tspan font-style="italic">R</tspan><tspan baseline-shift="sub" font-size="9">t</tspan></text>

    <!-- Weaver on the right: only H_t and R_t are conditioning inputs -->
    <rect id="right-weaver" x="1215" y="292" width="150" height="78" rx="9" class="module"/>
    <text x="1290" y="338" text-anchor="middle" class="label" font-weight="600">Weaver</text>
    <text x="1290" y="360" text-anchor="middle" class="small gray">(Frozen)</text>
    <path d="M930 317 V445 H1195 V331 H1215" class="control"/>
    <text x="952" y="437" class="math" text-anchor="middle"><tspan font-style="italic">H</tspan><tspan baseline-shift="sub" font-size="10">t</tspan></text>
    <rect id="right-mt" x="1262" y="405" width="56" height="34" rx="6" class="node"/>
    <text x="1290" y="428" class="label" text-anchor="middle" font-style="italic">m<tspan baseline-shift="sub" font-size="10">t</tspan></text>
    <path d="M1290 370 V405" class="latent"/>
    <!-- Independent m_t paths: blue injection and purple write-back -->
    <path d="M1290 439 V512 H826 V356" class="latent"/>
    <rect id="insert-update" x="1090" y="460" width="120" height="34" rx="7" class="action"/>
    <text x="1150" y="482" class="action-label" text-anchor="middle">Insert / Update</text>
    <path d="M1262 422 H1210 V477 H1210" class="memory" marker-end="url(#arrow-purple)"/>
    <path d="M1090 477 H1060 V435" class="memory" marker-end="url(#arrow-purple)"/>
  </g>
'''

# Preserve original defs, left panel, legend and caption; replace only right content.
start = src.index('  <!-- Right modules -->')
end = src.index('  <!-- Legend and footer -->')
out = src[:start] + right + src[end:]
out = out.replace('</style>', '.token-cell{fill:#f1eafa;stroke:#6b318f;stroke-width:1}.bank-title{font-family:Georgia,\'Times New Roman\',serif;font-size:14px;font-weight:600;fill:#572477}.slot-label{font-family:Georgia,\'Times New Roman\',serif;font-size:12px;fill:#572477}.ellipsis{font-size:14px;fill:#572477}.separator{stroke:#b595cb;stroke-width:1;stroke-dasharray:3 3}.bank-foot{font-family:Georgia,\'Times New Roman\',serif;font-size:10px;fill:#572477}.reset-box{fill:#fff;stroke:#6b318f;stroke-width:1.3;stroke-dasharray:5 3}.reset-text{font-family:Georgia,\'Times New Roman\',serif;font-size:11px;fill:#572477}.memory-label{font-family:Georgia,\'Times New Roman\',serif;font-size:12px;fill:#572477}.action{fill:#fcfaff;stroke:#6b318f;stroke-width:1.3}.action-label{font-family:Georgia,\'Times New Roman\',serif;font-size:13px;fill:#572477}</style>')
svg = root / 'figures/figure1_architecture_bank_left.svg'
png = root / 'figures/figure1_architecture_bank_left_preview.png'
svg.write_text(out, encoding='utf-8')
cairosvg.svg2png(bytestring=out.encode(), write_to=str(png), output_width=2800, output_height=1280)
print(svg); print(png)
