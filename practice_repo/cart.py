"""A tiny shopping-cart pricing module with planted bugs.

The notebook copies this folder into a scratch workspace and lets an agent fix it
until `pytest` passes. Do not fix the bugs here, or the labs have nothing to do.
"""

from dataclasses import dataclass, field

TAX_RATE = 0.09  # Singapore GST

DISCOUNT_CODES = {
    "SAVE10": ("percent", 10),   # 10% off the subtotal
    "FLAT5": ("flat", 5.00),     # $5 off the subtotal
}


@dataclass
class Item:
    name: str
    unit_price: float
    quantity: int = 1


@dataclass
class Cart:
    items: list[Item] = field(default_factory=list)
    discount_code: str | None = None

    def add(self, item: Item) -> None:
        if item.quantity <= 0:
            raise ValueError("quantity must be positive")
        self.items.append(item)

    def subtotal(self) -> float:
        # BUG 1: ignores quantity.
        return sum(i.unit_price for i in self.items)

    def discount(self) -> float:
        if self.discount_code is None:
            return 0.0
        kind, value = DISCOUNT_CODES[self.discount_code]
        if kind == "percent":
            # BUG 2: treats the percentage as dollars.
            return value
        return min(value, self.subtotal())

    def total(self) -> float:
        # BUG 3: tax is charged on the pre-discount amount, and the result is truncated, not rounded.
        taxed = self.subtotal() * (1 + TAX_RATE)
        return int((taxed - self.discount()) * 100) / 100
