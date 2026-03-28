"""Tests for card loading and registry."""

from startup_simulator.card_registry import get_registry, load_talent_cards, load_seed_products, load_growth_products
from startup_simulator.types import Tier, Sector, Tag, TalentType


def test_load_talent_cards():
    cards = load_talent_cards()
    assert len(cards) > 0
    # Should have juniors
    juniors = [c for c in cards if c.is_junior]
    assert len(juniors) == 2  # jr_software, jr_hardware
    # Should have seniors
    seniors = [c for c in cards if c.is_senior_dev]
    assert len(seniors) > 0


def test_load_seed_products():
    products = load_seed_products()
    assert len(products) == 30  # 20 core + 10 expansion
    # All should be Tier 1
    for p in products:
        assert p.tier == Tier.TIER1
    # Check no requirements for Tier 1
    for p in products:
        assert len(p.requires) == 0


def test_load_growth_products():
    products = load_growth_products()
    # Should have Tier 2, Tier 3, and Market Crash
    tiers = {p.tier for p in products}
    assert Tier.TIER2 in tiers
    assert Tier.TIER3 in tiers
    # Market Crash
    crash = [p for p in products if p.is_market_crash]
    assert len(crash) == 1


def test_registry_singleton():
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2


def test_registry_lookups():
    r = get_registry()
    # Talent lookup
    jr_sw = r.get_talent("jr_software")
    assert jr_sw.name == "Junior Software Dev"
    assert jr_sw.is_junior
    assert jr_sw.cost == 2

    # Product lookup
    todo = r.get_product("todo_list")
    assert todo.name == "To-Do List"
    assert todo.tier == Tier.TIER1
    assert todo.provides == Tag.APP

    # Growth product
    streaming = r.get_product("streaming_service")
    assert streaming.tier == Tier.TIER2
    assert Tag.SOCIAL in streaming.requires
    assert streaming.provides == Tag.MEDIA


def test_product_properties():
    r = get_registry()
    # Software-only
    todo = r.get_product("todo_list")
    assert todo.is_software_only
    assert not todo.is_hardware_only
    assert not todo.is_hybrid

    # Hardware-heavy hybrid
    thermostat = r.get_product("smart_thermostat")
    assert thermostat.is_hybrid

    # Pure hardware
    genetics = r.get_product("genetics_kit")
    assert genetics.is_hardware_only


def test_agency_deck():
    r = get_registry()
    # Agency deck should not include juniors
    for c in r.agency_deck_defs:
        assert not c.is_junior
    # Should have correct total (10 seniors + 12 specialists = 22)
    assert len(r.agency_deck_defs) == 22
