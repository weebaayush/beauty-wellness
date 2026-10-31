"""A tiny audit log: every cleaning action is recorded with the number of rows it touched."""
import json
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CleaningLog:
    entries: list = field(default_factory=list)
    shapes: list = field(default_factory=list)

    def add(self, step: str, action: str, rows_affected: int) -> None:
        self.entries.append({"step": step, "action": action, "rows_affected": int(rows_affected)})

    def shape(self, stage: str, products: int, reviews: int) -> None:
        self.shapes.append({"stage": stage, "products": int(products), "reviews": int(reviews)})

    def to_json(self, path, meta: dict | None = None) -> None:
        payload = {"generated": datetime.now().isoformat(timespec="seconds"), "meta": meta or {},
                   "shapes": self.shapes, "actions": self.entries}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def to_markdown(self) -> str:
        lines = ["| Step | Action | Rows affected |", "|---|---|---|"]
        lines += [f"| {e['step']} | {e['action']} | {e['rows_affected']:,} |" for e in self.entries]
        return "\n".join(lines)
