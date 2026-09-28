import hashlib
import hmac

from roostoo_bot.api_client import RoostooClient


def test_official_signature_example() -> None:
    secret = "S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep"
    params = {"timestamp": 1580774512000, "pair": "BNB/USD", "quantity": 2000, "side": "BUY", "type": "MARKET"}
    client = RoostooClient("key", secret)
    canonical = "pair=BNB/USD&quantity=2000&side=BUY&timestamp=1580774512000&type=MARKET"
    assert client.canonical_payload(params) == canonical
    assert client.sign(params) == hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    assert client.sign(params) == "20b7fd5550b67b3bf0c1684ed0f04885261db8fdabd38611e9e6af23c19b7fff"


def test_cancel_order_rejects_ambiguous_filters() -> None:
    client = RoostooClient("key", "secret")
    try:
        client.cancel_order(1, "BTC/USD")
    except ValueError as exc:
        assert "not both" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

