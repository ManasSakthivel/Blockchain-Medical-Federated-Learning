"""Solidity contract tests on the IN-MEMORY eth-tester EVM (py-evm).

This is NOT Ganache and NOT Truffle: it compiles the real contracts with solc
0.8.20 (via py-solc-x) and executes them on an in-process EVM. It validates
contract logic and access control; it does not validate Ganache/network
integration, gas pricing or the Truffle migration scripts.

Skipped automatically if solc 0.8.20 cannot be obtained (needs a one-time
download) or eth-tester is missing.
"""
import os

import pytest

solcx = pytest.importorskip("solcx")
pytest.importorskip("eth_tester")
from eth_tester.exceptions import TransactionFailed  # noqa: E402
from web3 import Web3  # noqa: E402
from web3.exceptions import ContractLogicError  # noqa: E402
from web3.providers.eth_tester import EthereumTesterProvider  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONTRACTS = os.path.join(ROOT, "contracts")
ZERO = "0x0000000000000000000000000000000000000000"


@pytest.fixture(scope="module")
def compiled():
    try:
        solcx.set_solc_version("0.8.20")
    except Exception:  # not installed and cannot download
        pytest.skip("solc 0.8.20 is not available")
    out = {}
    for name in ("ConsentContract", "FileVerificationContract", "ErasureLogContract",
                 "EHRContract", "FederatedLearning"):
        res = solcx.compile_files([os.path.join(CONTRACTS, f"{name}.sol")],
                                  output_values=["abi", "bin"], optimize=True,
                                  allow_paths=CONTRACTS)
        key = next(k for k in res if k.endswith(f":{name}"))
        out[name] = res[key]
    return out


@pytest.fixture()
def chain(compiled):
    w3 = Web3(EthereumTesterProvider())
    accts = w3.eth.accounts
    w3.eth.default_account = accts[0]

    def deploy(name, sender=None):
        c = w3.eth.contract(abi=compiled[name]["abi"], bytecode=compiled[name]["bin"])
        tx = c.constructor().transact({"from": sender or accts[0]})
        addr = w3.eth.wait_for_transaction_receipt(tx).contractAddress
        return w3.eth.contract(address=addr, abi=compiled[name]["abi"])

    class Chain:
        pass

    ch = Chain()
    ch.w3, ch.accts, ch.deploy = w3, accts, deploy
    return ch


def reverts(fn, *args, sender, match=None):
    """Call fn(*args) as `sender`; assert it reverts (optionally with `match`)."""
    with pytest.raises((ContractLogicError, TransactionFailed)) as exc:
        fn(*args).transact({"from": sender})
    if match:
        assert match in str(exc.value)


def digest(n):
    return bytes([n]) * 32


# ---------------------------------------------------------------------------
class TestConsentContract:
    def test_only_operator_can_write(self, chain):
        c = chain.deploy("ConsentContract")
        op, patient, rec, attacker = chain.accts[:4]
        assert c.functions.operator().call() == op
        reverts(c.functions.grantConsent, patient, rec, "file", digest(1), sender=attacker,
                match="not operator")
        # the PATIENT cannot grant or revoke for themselves: documented trust model
        reverts(c.functions.grantConsent, patient, rec, "file", digest(1), sender=patient,
                match="not operator")

    def test_grant_check_revoke(self, chain):
        c = chain.deploy("ConsentContract")
        op, patient, rec = chain.accts[:3]
        c.functions.grantConsent(patient, rec, "file", digest(1)).transact({"from": op})
        assert c.functions.hasConsent(patient, rec, "file").call() is True
        assert c.functions.hasConsent(patient, rec, "other").call() is False
        assert c.functions.getPatientConsentIds(patient).call() == [1]
        c.functions.revokeConsent(1).transact({"from": op})
        assert c.functions.hasConsent(patient, rec, "file").call() is False

    def test_regrant_supersedes_previous(self, chain):
        c = chain.deploy("ConsentContract")
        op, patient, rec = chain.accts[:3]
        c.functions.grantConsent(patient, rec, "file", digest(1)).transact({"from": op})
        c.functions.grantConsent(patient, rec, "file", digest(2)).transact({"from": op})
        assert c.functions.consents(1).call()[-1] is False   # first is inactive
        assert c.functions.consents(2).call()[-1] is True
        assert c.functions.hasConsent(patient, rec, "file").call() is True

    def test_validation(self, chain):
        c = chain.deploy("ConsentContract")
        op, patient, rec = chain.accts[:3]
        reverts(c.functions.grantConsent, ZERO, rec, "file", digest(1), sender=op)
        reverts(c.functions.grantConsent, patient, ZERO, "file", digest(1), sender=op)
        reverts(c.functions.grantConsent, patient, rec, "", digest(1), sender=op)
        reverts(c.functions.grantConsent, patient, rec, "x" * 65, digest(1), sender=op)
        reverts(c.functions.revokeConsent, 99, sender=op, match="not active")

    def test_double_revoke_reverts_and_operator_transfer(self, chain):
        c = chain.deploy("ConsentContract")
        op, patient, rec, new_op = chain.accts[:4]
        c.functions.grantConsent(patient, rec, "file", digest(1)).transact({"from": op})
        c.functions.revokeConsent(1).transact({"from": op})
        reverts(c.functions.revokeConsent, 1, sender=op, match="not active")
        reverts(c.functions.transferOperator, new_op, sender=new_op, match="not operator")
        reverts(c.functions.transferOperator, ZERO, sender=op)
        c.functions.transferOperator(new_op).transact({"from": op})
        reverts(c.functions.revokeConsent, 1, sender=op, match="not operator")

    def test_no_gdpr_or_erasure_surface(self, chain):
        c = chain.deploy("ConsentContract")
        names = {f["name"] for f in c.abi if f["type"] == "function"}
        assert not names & {"requestDataErasure", "requestDataPortability", "revokeAllConsents"}


class TestFileVerificationContract:
    def test_only_operator_and_no_metadata_stored(self, chain):
        c = chain.deploy("FileVerificationContract")
        op, attacker = chain.accts[:2]
        reverts(c.functions.anchorFile, digest(1), digest(2), sender=attacker, match="not operator")
        c.functions.anchorFile(digest(1), digest(2)).transact({"from": op})
        pub, dig, at, ver, mis = c.functions.getAnchor(1).call()
        assert pub == digest(1) and dig == digest(2) and at > 0 and (ver, mis) == (0, 0)
        # no personal / descriptive fields exist anywhere in the ABI
        names = " ".join(str(i["name"]) for f in c.abi for i in f.get("inputs", []))
        assert not any(w in names.lower() for w in ("filename", "patient", "ipfs", "filetype"))

    def test_duplicate_and_empty_rejected(self, chain):
        c = chain.deploy("FileVerificationContract")
        op = chain.accts[0]
        c.functions.anchorFile(digest(1), digest(2)).transact({"from": op})
        reverts(c.functions.anchorFile, digest(1), digest(3), sender=op, match="already anchored")
        reverts(c.functions.anchorFile, b"\x00" * 32, digest(3), sender=op)
        reverts(c.functions.anchorFile, digest(5), b"\x00" * 32, sender=op)

    def test_record_verification_counts(self, chain):
        c = chain.deploy("FileVerificationContract")
        op, attacker = chain.accts[:2]
        c.functions.anchorFile(digest(1), digest(2)).transact({"from": op})
        c.functions.recordVerification(1, True).transact({"from": op})
        c.functions.recordVerification(1, False).transact({"from": op})
        assert c.functions.getAnchor(1).call()[3:] == [2, 1]
        reverts(c.functions.recordVerification, 1, True, sender=attacker, match="not operator")
        reverts(c.functions.recordVerification, 9, True, sender=op, match="unknown file")


class TestErasureLogContract:
    def test_records_once_and_operator_only(self, chain):
        c = chain.deploy("ErasureLogContract")
        op, attacker = chain.accts[:2]
        reverts(c.functions.recordErasure, digest(1), sender=attacker, match="not operator")
        c.functions.recordErasure(digest(1)).transact({"from": op})
        assert c.functions.entryCount().call() == 1
        assert c.functions.recordedAt(digest(1)).call() > 0
        reverts(c.functions.recordErasure, digest(1), sender=op, match="already recorded")
        reverts(c.functions.recordErasure, b"\x00" * 32, sender=op)

    def test_no_person_field_in_abi(self, chain):
        c = chain.deploy("ErasureLogContract")
        for f in c.abi:
            for i in f.get("inputs", []):
                assert i["type"] in ("bytes32", "address", "uint256")
                assert i["type"] != "address" or f["name"] == "transferOperator" \
                    or f["name"] == "OperatorTransferred"


class TestEHRContract:
    def _setup(self, chain, with_consent=True):
        ehr = chain.deploy("EHRContract")
        consent = chain.deploy("ConsentContract")
        admin, doc, pat, other = chain.accts[:4]
        ehr.functions.addDoctor(doc, digest(1)).transact({"from": admin})
        ehr.functions.addPatient(pat, digest(2)).transact({"from": admin})
        ehr.functions.setConsentContract(consent.address).transact({"from": admin})
        if with_consent:
            consent.functions.grantConsent(pat, doc, "medical_record", digest(3)).transact(
                {"from": admin})
        return ehr, consent, admin, doc, pat, other

    def test_admin_only_registration(self, chain):
        ehr = chain.deploy("EHRContract")
        admin, a, b = chain.accts[:3]
        reverts(ehr.functions.addDoctor, a, digest(1), sender=b, match="Only admin")
        reverts(ehr.functions.addPatient, a, digest(1), sender=b, match="Only admin or doctor")
        reverts(ehr.functions.addDoctor, ZERO, digest(1), sender=admin)

    def test_record_requires_consent_and_fails_closed(self, chain):
        ehr, consent, admin, doc, pat, other = self._setup(chain, with_consent=False)
        reverts(ehr.functions.addMedicalRecord, pat, digest(9), "", sender=doc,
                match="not granted consent")
        consent.functions.grantConsent(pat, doc, "medical_record", digest(3)).transact(
            {"from": admin})
        ehr.functions.addMedicalRecord(pat, digest(9), "").transact({"from": doc})
        assert ehr.functions.getPatientRecords(pat).call() == [1]

    def test_gate_unset_fails_closed(self, chain):
        ehr = chain.deploy("EHRContract")
        admin, doc, pat = chain.accts[:3]
        ehr.functions.addDoctor(doc, digest(1)).transact({"from": admin})
        ehr.functions.addPatient(pat, digest(2)).transact({"from": admin})
        reverts(ehr.functions.addMedicalRecord, pat, digest(9), "", sender=doc,
                match="not configured")

    def test_recording_doctor_is_msg_sender_and_only_they_can_update(self, chain):
        ehr, consent, admin, doc, pat, other = self._setup(chain)
        ehr.functions.addMedicalRecord(pat, digest(9), "").transact({"from": doc})
        assert ehr.functions.medicalRecords(1).call()[2] == doc
        ehr.functions.addDoctor(other, digest(4)).transact({"from": admin})
        reverts(ehr.functions.updateMedicalRecord, 1, digest(8), "", sender=other,
                match="original doctor")
        ehr.functions.updateMedicalRecord(1, digest(8), "").transact({"from": doc})

    def test_update_blocked_after_consent_revoked(self, chain):
        ehr, consent, admin, doc, pat, other = self._setup(chain)
        ehr.functions.addMedicalRecord(pat, digest(9), "").transact({"from": doc})
        consent.functions.revokeConsent(1).transact({"from": admin})
        reverts(ehr.functions.updateMedicalRecord, 1, digest(8), "", sender=doc,
                match="Consent no longer active")

    def test_only_admin_invalidates(self, chain):
        ehr, consent, admin, doc, pat, other = self._setup(chain)
        ehr.functions.addMedicalRecord(pat, digest(9), "").transact({"from": doc})
        reverts(ehr.functions.invalidateRecord, 1, sender=doc, match="Only admin")
        ehr.functions.invalidateRecord(1).transact({"from": admin})
        reverts(ehr.functions.invalidateRecord, 1, sender=admin, match="already invalid")

    def test_no_plaintext_fields(self, chain):
        ehr = chain.deploy("EHRContract")
        names = {f["name"] for f in ehr.abi if f["type"] == "function"}
        assert not names & {"getPatient", "getDoctor", "doctors", "patients"}
        inputs = " ".join(i["name"] for f in ehr.abi for i in f.get("inputs", []))
        assert "medicalHistory" not in inputs and "licenseNumber" not in inputs

    def test_zero_address_never_has_role(self, chain):
        ehr = chain.deploy("EHRContract")
        assert ehr.functions.isDoctor(ZERO).call() is False
        assert ehr.functions.isPatient(ZERO).call() is False


class TestFederatedLearningContract:
    def _fl(self, chain):
        fl = chain.deploy("FederatedLearning")
        admin, p1, p2, outsider = chain.accts[:4]
        fl.functions.registerParticipant(p1).transact({"from": admin})
        fl.functions.registerParticipant(p2).transact({"from": admin})
        return fl, admin, p1, p2, outsider

    def test_outsider_cannot_record_update(self, chain):
        fl, admin, p1, p2, outsider = self._fl(chain)
        reverts(fl.functions.recordModelUpdate, digest(1), 1, 8000, sender=outsider,
                match="not authorised")
        fl.functions.recordModelUpdate(digest(1), 1, 8000).transact({"from": p1})
        assert fl.functions.getUpdateCount().call() == 1

    def test_bounds_enforced(self, chain):
        fl, admin, p1, p2, outsider = self._fl(chain)
        reverts(fl.functions.recordModelUpdate, digest(1), 1, 10001, sender=p1,
                match="out of range")
        reverts(fl.functions.recordModelUpdate, digest(1), 50, 8000, sender=p1,
                match="too far ahead")

    def test_only_admin_registers_and_no_staking(self, chain):
        fl, admin, p1, p2, outsider = self._fl(chain)
        reverts(fl.functions.registerParticipant, outsider, sender=outsider)
        names = {f["name"] for f in fl.abi if f["type"] == "function"}
        assert "stake" not in names

    def test_round_summary_aggregator_only_and_flag_deactivates(self, chain):
        fl, admin, p1, p2, outsider = self._fl(chain)
        reverts(fl.functions.recordRoundSummary, digest(1), 1, 8000, [p2], sender=p1,
                match="not aggregator")
        reverts(fl.functions.recordRoundSummary, digest(1), 1, 8000, [outsider], sender=admin,
                match="not registered")
        fl.functions.recordRoundSummary(digest(1), 1, 8000, [p2]).transact({"from": admin})
        # flagged participant is deactivated and can no longer write
        reverts(fl.functions.recordModelUpdate, digest(2), 2, 8000, sender=p2,
                match="not authorised")
        reverts(fl.functions.recordRoundSummary, digest(1), 0, 8000, [], sender=admin,
                match="backwards")

    def test_reputation_ema(self, chain):
        fl, admin, p1, p2, outsider = self._fl(chain)
        fl.functions.recordModelUpdate(digest(1), 1, 9000).transact({"from": p1})
        assert fl.functions.getReputation(p1).call() == (7 * 9000 + 3 * 8000) // 10

    def test_aggregator_can_record_on_behalf_of_registered_participant(self, chain):
        fl, admin, p1, p2, outsider = self._fl(chain)
        # Aggregator records on behalf of registered participant p1
        fl.functions.recordModelUpdate(digest(1), 1, 9000, p1).transact({"from": admin})
        assert fl.functions.getUpdateCount().call() == 1
        u = fl.functions.getUpdate(0).call()
        assert u[0] == p1  # contributor is participant p1, not admin/aggregator
        assert u[1] == digest(1)
        assert u[2] == 1
        assert u[5] == 9000
        # Participant reputation updated via EMA
        assert fl.functions.getReputation(p1).call() == (7 * 9000 + 3 * 8000) // 10
        # Aggregator recording with ZERO defaults to self-attribution (simulation fallback)
        fl.functions.recordModelUpdate(digest(2), 1, 8500, ZERO).transact({"from": admin})
        assert fl.functions.getUpdateCount().call() == 2
        assert fl.functions.getUpdate(1).call()[0] == admin

    def test_aggregator_on_behalf_of_unregistered_address_reverts(self, chain):
        fl, admin, p1, p2, outsider = self._fl(chain)
        reverts(fl.functions.recordModelUpdate, digest(1), 1, 8000, outsider, sender=admin,
                match="not registered")

    def test_participant_records_own_update(self, chain):
        fl, admin, p1, p2, outsider = self._fl(chain)
        # Participant records directly using 3-arg overload
        fl.functions.recordModelUpdate(digest(1), 1, 8500).transact({"from": p1})
        u = fl.functions.getUpdate(0).call()
        assert u[0] == p1
        assert u[1] == digest(1)
        assert u[2] == 1
        assert u[5] == 8500
        # Participant records using 4-arg overload with their own address or ZERO
        fl.functions.recordModelUpdate(digest(2), 1, 9000, p1).transact({"from": p1})
        assert fl.functions.getUpdate(1).call()[0] == p1
        fl.functions.recordModelUpdate(digest(3), 1, 9000, ZERO).transact({"from": p1})
        assert fl.functions.getUpdate(2).call()[0] == p1
        # Participant cannot record on behalf of another participant
        reverts(fl.functions.recordModelUpdate, digest(4), 1, 8000, p2, sender=p1,
                match="cannot act on behalf")

    def test_unauthorized_caller_rejected(self, chain):
        fl, admin, p1, p2, outsider = self._fl(chain)
        reverts(fl.functions.recordModelUpdate, digest(1), 1, 8000, sender=outsider,
                match="not authorised")
        reverts(fl.functions.recordModelUpdate, digest(1), 1, 8000, outsider, sender=outsider,
                match="not authorised")
        reverts(fl.functions.recordModelUpdate, digest(1), 1, 8000, p1, sender=outsider,
                match="not authorised")
        reverts(fl.functions.recordModelUpdate, digest(1), 1, 8000, ZERO, sender=outsider,
                match="not authorised")


# ---------------------------------------------------------------------------
# TestRolesLibrary — covers the two Agent B changes to Roles.sol
#
# The Roles library uses `internal` functions, so it is exercised through the
# contracts that embed it (EHRContract for doctor/patient roles, FederatedLearning
# for the admin/participant roles).
#
# Change 1 — add(): require(account != address(0)) was added.
#   Before: add(address(0)) would silently set bearer[0x00] = true.
#   After:  add(address(0)) reverts with "Roles: account is the zero address".
#
# Change 2 — has(): require(...zero...) was replaced with `if (...) return false`.
#   Before: has(address(0)) reverted.
#   After:  has(address(0)) returns false (safe for view-only callers).
# ---------------------------------------------------------------------------
class TestRolesLibrary:
    def test_add_zero_address_reverts(self, chain):
        """Agent B change 1: Roles.add() must revert when given address(0).

        EHRContract.addDoctor internally calls doctor.add(drId), so passing
        address(0) as the doctor id should now revert rather than silently
        recording the zero address as a valid doctor.
        """
        ehr = chain.deploy("EHRContract")
        admin = chain.accts[0]
        # address(0) must be rejected
        reverts(ehr.functions.addDoctor, ZERO, digest(1), sender=admin,
                match="zero address")
        # same guard via FederatedLearning.registerParticipant → participantRole.add()
        fl = chain.deploy("FederatedLearning")
        reverts(fl.functions.registerParticipant, ZERO, sender=admin,
                match="zero address")

    def test_has_zero_address_returns_false_not_revert(self, chain):
        """Agent B change 2: Roles.has() must return false for address(0).

        Before the fix, has(address(0)) called `require(account != address(0))`
        which reverted. Now it returns false so view helpers (isDoctor, isPatient,
        isParticipant) are safe to call with the zero address.
        """
        ehr = chain.deploy("EHRContract")
        fl = chain.deploy("FederatedLearning")
        # These are view calls — they must NOT revert; they must return False.
        assert ehr.functions.isDoctor(ZERO).call() is False
        assert ehr.functions.isPatient(ZERO).call() is False
        assert fl.functions.isParticipant(ZERO).call() is False

    def test_add_and_has_roundtrip_with_legitimate_address(self, chain):
        """Sanity check: the Agent B changes do not break normal Roles usage.

        A legitimate (non-zero) address can still be added to a role and
        has() returns True for it, False before it is added, and False again
        after it is removed (via FederatedLearning's deactivation path).
        """
        ehr = chain.deploy("EHRContract")
        admin, doc = chain.accts[:2]
        # Before add: has() must return False (not revert)
        assert ehr.functions.isDoctor(doc).call() is False
        # After add: has() must return True
        ehr.functions.addDoctor(doc, digest(5)).transact({"from": admin})
        assert ehr.functions.isDoctor(doc).call() is True
        # Duplicate add must still revert (the existing guard is unchanged)
        reverts(ehr.functions.addDoctor, doc, digest(5), sender=admin,
                match="already has role")
