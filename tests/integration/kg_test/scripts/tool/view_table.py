import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

from src.utils.json_utils import loads

with open("tests/integration/kg_test/33.json", "rb") as f:
    data = loads(f.read())

lines = []
lines.append("=" * 80)
lines.append(f"总实体: {len(data['entities'])}  总关系: {len(data['relations'])}")
lines.append("=" * 80)
lines.append("")

lines.append("━" * 80)
lines.append("  ENTITIES")
lines.append("━" * 80)
lines.append(f"{'序号':>4s}  {'name':30s}  {'type':10s}  {'type_en':15s}  {'attributes'}")
lines.append("━" * 80)
for i, e in enumerate(data["entities"], 1):
    attrs = ", ".join(f"{k}={v}" for k, v in e.get("attributes", {}).items()) if e.get("attributes") else ""
    lines.append(f"{i:4d}  {e['name']:30s}  {e['type']:10s}  {e['type_en']:15s}  {attrs[:50]}")

lines.append("")
lines.append("━" * 80)
lines.append("  RELATIONS")
lines.append("━" * 80)
lines.append(f"{'序号':>4s}  {'from_entity':30s}  {'type':12s}  {'type_en':15s}  {'to_entity'}")
lines.append("━" * 80)
for i, r in enumerate(data["relations"], 1):
    lines.append(f"{i:4d}  {r['from_entity']:30s}  {r['type']:12s}  {r['type_en']:15s}  {r['to_entity']}")
lines.append("━" * 80)

output_path = Path(__file__).resolve().parent / "view_table.txt"
output_path.write_text("\n".join(lines), encoding="utf-8")
print(f"已输出到: {output_path}")
