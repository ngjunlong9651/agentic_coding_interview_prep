import pytest

from cart import Cart, Item


def test_subtotal_uses_quantity():
    cart = Cart([Item("pen", 2.50, quantity=4)])
    assert cart.subtotal() == pytest.approx(10.00)


def test_percent_discount():
    cart = Cart([Item("book", 40.00)], discount_code="SAVE10")
    assert cart.discount() == pytest.approx(4.00)


def test_flat_discount_never_exceeds_subtotal():
    cart = Cart([Item("sticker", 3.00)], discount_code="FLAT5")
    assert cart.discount() == pytest.approx(3.00)


def test_total_applies_tax_after_discount_and_rounds():
    # (40 - 4) * 1.09 = 39.24
    cart = Cart([Item("book", 40.00)], discount_code="SAVE10")
    assert cart.total() == pytest.approx(39.24)


def test_total_rounds_not_truncates():
    # 1.55 * 1.09 = 1.6895. Rounding gives 1.69; truncating would give 1.68.
    cart = Cart([Item("gum", 1.55)])
    assert cart.total() == pytest.approx(1.69)


def test_unknown_code_raises():
    cart = Cart([Item("book", 40.00)], discount_code="NOPE")
    with pytest.raises(KeyError):
        cart.total()


def test_rejects_non_positive_quantity():
    with pytest.raises(ValueError):
        Cart().add(Item("ghost", 1.00, quantity=0))
