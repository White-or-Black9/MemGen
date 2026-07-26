from pathlib import Path
import cairosvg


root = Path(__file__).parent
source = (root / "figures/figure1_architecture.svg").read_text(encoding="utf-8")


def token_cells() -> str:
    return "".join(
        f'<rect x="{920 + 14*i}" y="414" width="11" height="17" rx="1.4" class="token-cell"/>'
        for i in range(8)
    )


right_panel = f'''  <!-- Right panel: refined Bank-below / Weaver-right layout -->
  <g id="right-panel">
    <!-- Primary modules: aligned reasoning and generation row -->
    <rect id="right-reasoner" x="760" y="235" width="164" height="78" rx="9" class="module"/>
    <text x="842" y="267" text-anchor="middle" class="label" font-weight="600">Reasoner</text>
    <text x="842" y="290" text-anchor="middle" class="small">(Frozen)</text>
    <rect id="right-weaver" x="1188" y="235" width="164" height="78" rx="9" class="module"/>
    <text x="1270" y="267" text-anchor="middle" class="label" font-weight="600">Weaver</text>
    <text x="1270" y="290" text-anchor="middle" class="small">(Frozen)</text>
    <rect id="right-trigger" x="978" y="105" width="150" height="74" rx="9" class="module"/>
    <text x="1053" y="136" text-anchor="middle" class="label" font-weight="600">Trigger</text>
    <text x="1053" y="158" text-anchor="middle" class="small">(Frozen)</text>

    <!-- Direct H_t main path, with distinct branches to Trigger and Bank -->
    <path d="M924 274 H1188" class="control"/>
    <rect x="1023" y="249" width="54" height="21" rx="5" class="label-bg"/>
    <text x="1050" y="264" text-anchor="middle" class="math-label" font-style="italic">Hₜ</text>
    <path d="M950 274 V198 H1053 V179" class="control"/>
    <path d="M1030 274 V355 H977 V380" class="memory"/>
    <rect x="900" y="327" width="124" height="23" rx="5" class="label-bg"/>
    <text x="976" y="343" text-anchor="end" class="memory-label">Query with</text>
    <text x="981" y="343" class="memory-label" font-style="italic">Hₜ</text>

    <!-- Trigger decisions: separate dashed control paths -->
    <path d="M978 142 H800 V235 H842" class="control-dashed"/>
    <rect x="812" y="120" width="55" height="22" rx="5" class="label-bg"/>
    <text x="839" y="136" text-anchor="middle" class="small gray">SKIP</text>
    <path d="M1128 142 H1270 V235" class="control-dashed"/>
    <rect x="1141" y="120" width="82" height="22" rx="5" class="label-bg"/>
    <text x="1182" y="136" text-anchor="middle" class="small gray">INVOKE</text>

    <!-- Compressed horizontal session-local memory bank -->
    <rect id="memory-bank" x="842" y="380" width="266" height="92" rx="11" class="bank"/>
    <text x="975" y="402" text-anchor="middle" class="bank-title">Session-Local Latent Memory Bank</text>
    <text x="869" y="427" class="slot-label">Slot i</text>{token_cells()}
    <text x="869" y="449" class="bank-note">× N bounded slots</text>
    <text x="1083" y="463" text-anchor="end" class="bank-foot">Capacity-aware replacement</text>
    <rect id="reset-session" x="740" y="340" width="128" height="25" rx="6" class="reset-box"/>
    <text x="804" y="357" text-anchor="middle" class="reset-text">Reset at session end</text>
    <path d="M868 352 H890 V380" class="memory-dashed"/>

    <!-- R_t retrieval: a short, visually separate lower input to Weaver -->
    <path d="M1108 425 H1150 V294 H1188" class="memory"/>
    <rect x="1035" y="326" width="142" height="23" rx="5" class="label-bg"/>
    <text x="1137" y="342" text-anchor="end" class="memory-label">Retrieve top-k</text>
    <text x="1142" y="342" class="memory-label" font-style="italic">Rₜ</text>

    <!-- Explicit m_t node and two independent outputs -->
    <rect id="right-mt" x="1242" y="325" width="56" height="34" rx="6" class="node"/>
    <text x="1270" y="348" class="label" text-anchor="middle" font-style="italic">m<tspan baseline-shift="sub" font-size="10">t</tspan></text>
    <path d="M1270 313 V325" class="latent"/>
    <path d="M1270 359 V500 H800 V330 H842 V313" class="latent"/>
    <rect id="insert-update" x="1135" y="400" width="120" height="34" rx="7" class="action"/>
    <text x="1195" y="422" class="action-label" text-anchor="middle">Insert / Update</text>
    <path d="M1242 342 H1270 V417 H1255" class="memory"/>
    <path d="M1135 417 H1108" class="memory"/>
  </g>
'''

start = source.index("  <!-- Right modules -->")
end = source.index("  <!-- Legend and footer -->")
out = source[:start] + right_panel + source[end:]
extra_css = '''
      .label-bg{fill:#ffffff;stroke:none;}
      .math-label{font-family:Georgia,'Times New Roman',serif;font-size:15px;fill:#202b3a;}
      .token-cell{fill:#f1eafa;stroke:#6b318f;stroke-width:1;}
      .bank-title{font-family:Georgia,'Times New Roman',serif;font-size:14px;font-weight:600;fill:#572477;}
      .slot-label{font-family:Georgia,'Times New Roman',serif;font-size:12px;fill:#572477;}
      .bank-note{font-family:Georgia,'Times New Roman',serif;font-size:10.5px;fill:#725782;}
      .bank-foot{font-family:Georgia,'Times New Roman',serif;font-size:9.5px;fill:#572477;}
      .reset-box{fill:#ffffff;stroke:#6b318f;stroke-width:1.2;stroke-dasharray:5 3;}
      .reset-text{font-family:Georgia,'Times New Roman',serif;font-size:10.5px;fill:#572477;}
      .memory-label{font-family:Georgia,'Times New Roman',serif;font-size:11.5px;fill:#572477;}
      .action{fill:#fcfaff;stroke:#6b318f;stroke-width:1.3;}
      .action-label{font-family:Georgia,'Times New Roman',serif;font-size:12.5px;fill:#572477;}
'''
out = out.replace("    </style>", extra_css + "    </style>")

svg = root / "figures/figure1_architecture_refined.svg"
preview = root / "figures/figure1_architecture_refined_preview.png"
svg.write_text(out, encoding="utf-8")
cairosvg.svg2png(bytestring=out.encode("utf-8"), write_to=str(preview), output_width=2800, output_height=1280)
print(svg)
print(preview)
