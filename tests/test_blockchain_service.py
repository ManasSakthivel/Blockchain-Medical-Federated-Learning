"""BlockchainService / IPFSService / anchoring tests.

Chain tests run on the in-memory eth-tester EVM (NOT Ganache). The operator key
is generated per test run; no key material is stored in the repository.
IPFS tests use a fake HTTP session: they verify our client logic, not a real node.
"""
import os

import pytest
import requests

from app.services import anchoring
from app.services.blockchain_service import (BlockchainService, BlockchainUnavailable,
                                              ChainResult, _bytes32, address_from_did)
from app.services.ipfs_service import IPFSService, IPFSUnavailableError

solcx = pytest.importorskip("solcx")
pytest.importorskip("eth_tester")
from eth_account import Account  # noqa: E402
from web3 import Web3  # noqa: E402
from web3.providers.eth_tester import EthereumTesterProvider  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
NAMES = ("ConsentContract", "FileVerificationContract", "ErasureLogContract")


@pytest.fixture(scope="module")
def compiled():
    try:
        solcx.set_solc_version("0.8.20")
    except Exception:
        pytest.skip("solc 0.8.20 is not available")
    out = {}
    for n in NAMES:
        res = solcx.compile_files([os.path.join(ROOT, "contracts", f"{n}.sol")],
                                  output_values=["abi", "bin"], optimize=True,
                                  allow_paths=os.path.join(ROOT, "contracts"))
        out[n] = res[next(k for k in res if k.endswith(f":{n}"))]
    return out


@pytest.fixture()
def svc(compiled):
    w3 = Web3(EthereumTesterProvider())
    operator = Account.create()
    w3.eth.send_transaction({"from": w3.eth.accounts[0], "to": operator.address,
                             "value": w3.to_wei(10, "ether")})
    contracts = {}
    for n in NAMES:
        c = w3.eth.contract(abi=compiled[n]["abi"], bytecode=compiled[n]["bin"])
        # deploy FROM the operator so it becomes the contract operator
        tx = c.constructor().build_transaction({
            "from": operator.address, "nonce": w3.eth.get_transaction_count(operator.address),
            "chainId": w3.eth.chain_id, "gasPrice": w3.eth.gas_price})
        signed = operator.sign_transaction(tx)
        addr = w3.eth.wait_for_transaction_receipt(
            w3.eth.send_raw_transaction(signed.raw_transaction)).contractAddress
        contracts[n] = w3.eth.contract(address=addr, abi=compiled[n]["abi"])
    return BlockchainService(w3, contracts, operator.key.hex())


RECIPIENT = "0x" + "ab" * 20
DID = "did:ethr:" + Account.create().address
HEX32 = "11" * 32


class TestHelpers:
    def test_bytes32_validation(self):
        assert len(_bytes32("0x" + "aa" * 32, name="x")) == 32
        assert _bytes32("ab", name="x") == b"\xab" + b"\x00" * 31
        for bad in ("", "zz", "aa" * 33):
            with pytest.raises(ValueError):
                _bytes32(bad, name="x")

    def test_address_from_did(self):
        addr = Account.create().address
        assert address_from_did(f"did:ethr:{addr}") == addr
        with pytest.raises(ValueError):
            address_from_did("did:ethr:not-an-address")


class TestChainService:
    def test_anchor_file_and_read_back(self, svc):
        res = svc.anchor_file("a" * 32, HEX32)
        assert res.status == "anchored" and res.ok and res.object_id == 1
        assert res.tx_hash.startswith("0x")
        anchor = svc.get_file_anchor(1)
        assert anchor["digest"] == HEX32 and anchor["mismatches"] == 0

    def test_duplicate_anchor_reports_failed_not_anchored(self, svc):
        assert svc.anchor_file("a" * 32, HEX32).status == "anchored"
        dup = svc.anchor_file("a" * 32, HEX32)
        assert dup.status == "failed" and not dup.ok

    def test_verification_result_is_recorded_not_assumed(self, svc):
        svc.anchor_file("b" * 32, HEX32)
        ok = svc.record_file_verification(1, True)
        bad = svc.record_file_verification(1, False)
        assert (ok.status, ok.match) == ("anchored", True)
        assert (bad.status, bad.match) == ("anchored", False)   # not hard-coded True
        assert svc.get_file_anchor(1)["mismatches"] == 1
        assert svc.record_file_verification(99, True).status == "failed"

    def test_consent_lifecycle(self, svc):
        g = svc.grant_consent(DID, RECIPIENT, "file", "0x" + HEX32)
        assert g.status == "anchored" and g.object_id == 1
        assert svc.has_consent(DID, RECIPIENT, "file") is True
        r = svc.revoke_consent(g.object_id)
        assert r.status == "anchored"
        assert svc.has_consent(DID, RECIPIENT, "file") is False
        assert svc.revoke_consent(g.object_id).status == "failed"   # already revoked

    def test_erasure_log(self, svc):
        assert svc.record_erasure("cd" * 16).status == "anchored"
        assert svc.record_erasure("cd" * 16).status == "failed"     # duplicate ref

    def test_bad_input_is_failed_not_exception(self, svc):
        assert svc.grant_consent("did:ethr:nope", RECIPIENT, "file", HEX32).status == "failed"
        assert svc.anchor_file("zz", HEX32).status == "failed"
        assert svc.record_erasure("").status == "failed"

    def test_non_operator_wallet_cannot_write(self, svc, compiled):
        """A service signed with a different key is rejected by the contract."""
        w3 = svc.w3
        other = Account.create()
        w3.eth.send_transaction({"from": w3.eth.accounts[0], "to": other.address,
                                 "value": w3.to_wei(1, "ether")})
        imposter = BlockchainService(w3, svc.contracts, other.key.hex())
        assert imposter.anchor_file("c" * 32, HEX32).status == "failed"

    def test_status_has_no_secrets(self, svc):
        s = svc.status()
        assert s["connected"] is True and s["operator"] == svc.operator
        assert svc._key not in repr(s)

    def test_unreachable_rpc_is_failed_not_raised(self, svc):
        svc.w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:9", request_kwargs={"timeout": 1}))
        assert svc.anchor_file("d" * 32, HEX32).status == "failed"


class TestFromEnv:
    def test_unconfigured_is_unavailable(self, monkeypatch):
        monkeypatch.delenv("BLOCKCHAIN_RPC_URL", raising=False)
        monkeypatch.delenv("BLOCKCHAIN_OPERATOR_PRIVATE_KEY", raising=False)
        with pytest.raises(BlockchainUnavailable):
            BlockchainService.from_env()

    def test_key_without_rpc_is_unavailable(self, monkeypatch):
        monkeypatch.delenv("BLOCKCHAIN_RPC_URL", raising=False)
        monkeypatch.setenv("BLOCKCHAIN_OPERATOR_PRIVATE_KEY", "0x" + "11" * 32)
        with pytest.raises(BlockchainUnavailable):
            BlockchainService.from_env()

    def test_unreachable_rpc_is_unavailable(self, monkeypatch):
        monkeypatch.setenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:9")
        monkeypatch.setenv("BLOCKCHAIN_OPERATOR_PRIVATE_KEY", "0x" + "11" * 32)
        with pytest.raises(BlockchainUnavailable):
            BlockchainService.from_env()

    def test_no_private_keys_json_fallback_exists(self):
        src = open(os.path.join(ROOT, "app", "services", "blockchain_service.py")).read()
        assert "private_keys.json" not in src and "print(" not in src


class TestAnchoringFacade:
    """The app must keep working (with honest statuses) when the chain is absent or broken."""

    def test_unavailable_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("BLOCKCHAIN_RPC_URL", raising=False)
        assert anchoring.anchor_file("a" * 32, HEX32).status == "unavailable"
        assert anchoring.anchor_consent(1, DID, RECIPIENT, "file", HEX32).status == "unavailable"
        assert anchoring.revoke_consent_anchor(1).status == "unavailable"
        assert anchoring.record_erasure("ab" * 16).status == "unavailable"
        assert anchoring.record_verification(1, True).status == "unavailable"
        assert anchoring.unpin("Qm") is False

    def test_service_exception_becomes_failed(self, monkeypatch):
        class Boom:
            def anchor_file(self, *a):
                raise RuntimeError("rpc exploded")

        monkeypatch.setattr(anchoring, "_service", lambda: Boom())
        assert anchoring.anchor_file("a" * 32, HEX32).status == "failed"

    def test_successful_result_is_passed_through(self, monkeypatch):
        class Ok:
            def anchor_file(self, *a):
                return ChainResult("anchored", "0xabc", 7)

        monkeypatch.setattr(anchoring, "_service", lambda: Ok())
        out = anchoring.anchor_file("a" * 32, HEX32)
        assert (out.status, out.tx_hash, out.object_id) == ("anchored", "0xabc", 7)


class _Resp:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code, self._payload, self.content = status, payload, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Session:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        if self.exc:
            raise self.exc
        return self.resp


class TestIPFS:
    def test_add_returns_real_cid_and_pins(self):
        s = _Session(_Resp(payload={"Hash": "QmReal"}))
        assert IPFSService("http://ipfs:5001/", s).add_bytes(b"ciphertext") == "QmReal"
        url, kw = s.calls[0]
        assert url == "http://ipfs:5001/api/v0/add" and kw["params"]["pin"] == "true"

    def test_unreachable_raises_and_never_returns_a_mock_hash(self):
        s = _Session(exc=requests.ConnectionError("down"))
        with pytest.raises(IPFSUnavailableError):
            IPFSService("http://x", s).add_bytes(b"x")
        src = open(os.path.join(ROOT, "app", "services", "ipfs_service.py")).read()
        assert "QmMOCK".upper() not in src.upper()
        assert "_mock_hash" not in src

    def test_http_error_and_bad_payload(self):
        with pytest.raises(IPFSUnavailableError):
            IPFSService("http://x", _Session(_Resp(status=500))).add_bytes(b"x")
        with pytest.raises(IPFSUnavailableError):
            IPFSService("http://x", _Session(_Resp(payload={"oops": 1}))).add_bytes(b"x")

    def test_unpin_and_health_and_get(self):
        assert IPFSService("http://x", _Session(_Resp())).unpin("Qm") is True
        assert IPFSService("http://x", _Session(exc=requests.Timeout())).unpin("Qm") is False
        assert IPFSService("http://x", _Session(_Resp())).health() is True
        assert IPFSService("http://x", _Session(exc=requests.Timeout())).health() is False
        assert IPFSService("http://x", _Session(_Resp(content=b"abc"))).get_bytes("Qm") == b"abc"

    def test_from_env_requires_url(self, monkeypatch):
        monkeypatch.delenv("IPFS_API_URL", raising=False)
        with pytest.raises(IPFSUnavailableError):
            IPFSService.from_env()
        monkeypatch.setenv("IPFS_API_URL", "http://ipfs:5001")
        assert IPFSService.from_env().url == "http://ipfs:5001"
