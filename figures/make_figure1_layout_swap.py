from pathlib import Path
import cairosvg

base = Path(__file__).parent
s = (base / 'figure1_architecture_aaai_v2.svg').read_text(encoding='utf-8')

# Swap the two principal right-panel components: Weaver moves to the former
# bank position; the full editable bank group moves to the former Weaver area.
s = s.replace('<rect x="990" y="352" width="150" height="78" rx="9" class="module"/>', '<rect x="1210" y="174" width="150" height="78" rx="9" class="module"/>')
s = s.replace('<text x="1065" y="384" text-anchor="middle" class="label" font-weight="600">Weaver</text>', '<text x="1285" y="206" text-anchor="middle" class="label" font-weight="600">Weaver</text>')

start = s.index('  <!-- Memory bank -->')
stop = s.index('  <!-- Query/retrieval have only their prescribed endpoints -->')
bank = s[start:stop]
bank = bank.replace('  <!-- Memory bank -->', '  <!-- Memory bank (swapped to Weaver position) -->\n  <g transform="translate(-220 178)">') + '  </g>\n'
s = s[:start] + bank + s[stop:]

# Reroute the three connections to their swapped destinations.
s = s.replace('M930 317 L960 317 L960 391 L990 391', 'M930 317 L1150 317 L1150 213 L1210 213')
s = s.replace('M1100 135 L1180 135 L1180 330 L1065 330 L1065 352', 'M1100 135 L1180 135 L1180 213 L1210 213')
s = s.replace('M930 317 L1170 317 L1170 272 L1210 272', 'M930 317 L990 317 L990 450')
s = s.replace('M1210 332 L1165 332 L1165 391 L1140 391', 'M1144 460 L1170 460 L1170 213 L1210 213')

out = base / 'figure1_architecture_layout_swapped.svg'
preview = base / 'figure1_architecture_layout_swapped_preview.png'
out.write_text(s, encoding='utf-8')
cairosvg.svg2png(bytestring=s.encode(), write_to=str(preview), output_width=2800, output_height=1280)
print(out)
