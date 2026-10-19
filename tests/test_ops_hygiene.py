"""Regression tests for operator tooling: CLI, migrations, repository hygiene."""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _tracked_files():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    return [f for f in out.stdout.splitlines()
            if os.path.exists(os.path.join(ROOT, f))]


# ---------------------------------------------------------------------------
# run.py / create-user CLI
# ---------------------------------------------------------------------------
@pytest.fixture()
def runner(app):
    import run  # noqa: F401  (registers the CLI command on its own app)
    return run.app.test_cli_runner()


def test_run_py_has_no_hardcoded_credentials_or_debug():
    src = _read("run.py")
    assert "admin123" not in src
    assert "debug=True" not in src
    assert "0.0.0.0" not in src


def test_create_user_cli_prompts_for_password_and_hashes(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'cli.db'}")
    monkeypatch.setenv("EHR_DEV_MODE", "1")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "up"))
    code = (
        "import run\n"
        "from app import db\n"
        "from app.models import User\n"
        "r = run.app.test_cli_runner()\n"
        "ok = r.invoke(args=['create-user','--role','admin','--email','Boss@Example.org'],"
        " input='Correct-Horse-9\\nCorrect-Horse-9\\n')\n"
        "print('EXIT', ok.exit_code)\n"
        "weak = r.invoke(args=['create-user','--role','admin','--email','b@example.org'],"
        " input='short\\nshort\\n')\n"
        "print('WEAK', weak.exit_code)\n"
        "dup = r.invoke(args=['create-user','--role','admin','--email','boss@example.org'],"
        " input='Correct-Horse-9\\nCorrect-Horse-9\\n')\n"
        "print('DUP', dup.exit_code)\n"
        "with run.app.app_context():\n"
        "    u = User.query.one()\n"
        "    print('ROLE', u.role, u.email, u.check_password('Correct-Horse-9'),"
        " 'Correct-Horse-9' in u.password_hash)\n"
    )
    res = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT,
                         capture_output=True, text=True, env=os.environ.copy())
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "EXIT 0" in out
    assert "WEAK 1" in out          # short password rejected
    assert "DUP 1" in out           # duplicate e-mail (case-insensitive) rejected
    assert "ROLE admin boss@example.org True False" in out


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------
def test_migrations_build_the_full_schema_with_no_drift(tmp_path):
    env = dict(os.environ, DATABASE_URL=f"sqlite:///{tmp_path/'m.db'}",
               SECRET_KEY="x" * 40, FLASK_APP="run.py", UPLOAD_DIR=str(tmp_path / "u"))
    env.pop("EHR_DEV_MODE", None)   # prove create_all is not what builds the schema
    up = subprocess.run([sys.executable, "-W", "ignore", "-m", "flask", "db", "upgrade",
                         "--directory", "db_migrations"],
                        cwd=ROOT, capture_output=True, text=True, env=env)
    assert up.returncode == 0, up.stderr[-800:]
    chk = subprocess.run([sys.executable, "-W", "ignore", "-m", "flask", "db", "check",
                          "--directory", "db_migrations"],
                         cwd=ROOT, capture_output=True, text=True, env=env)
    assert chk.returncode == 0, chk.stderr[-800:]


# ---------------------------------------------------------------------------
# Repository hygiene
# ---------------------------------------------------------------------------
_PRIVKEY_ASSIGN = re.compile(
    r"""(?ix)(private[_-]?key|secret|password)\w*\s*[:=]\s*['"]?(0x)?[0-9a-f]{64}['"]?""")


def test_no_hex_private_key_assignments_in_tracked_text_files():
    offenders = []
    for rel in _tracked_files():
        if not rel.endswith((".py", ".js", ".json", ".yml", ".yaml", ".md", ".txt",
                             ".example", ".sol", ".html", ".toml", ".cfg", ".ini")):
            continue
        if rel in ("package-lock.json",) or rel.startswith("tests/"):
            continue
        try:
            text = _read(rel)
        except (UnicodeDecodeError, OSError):
            continue
        if _PRIVKEY_ASSIGN.search(text):
            offenders.append(rel)
    assert offenders == [], f"possible hard-coded private keys in: {offenders}"


def test_env_example_contains_no_secret_values():
    for line in _read(".env.example").splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name in ("SECRET_KEY", "FILE_KEK", "ANCHOR_HMAC_KEY",
                    "BLOCKCHAIN_OPERATOR_PRIVATE_KEY"):
            assert value.strip() == "", f"{name} must be empty in .env.example"


def test_compose_publishes_ports_on_loopback_only_and_has_no_secrets():
    text = "\n".join(line for line in _read("docker-compose.yml").splitlines()
                     if not line.lstrip().startswith("#"))
    ports = re.findall(r'^\s*-\s*"([^"]+:\d+:\d+)"', text, flags=re.M)
    assert ports, "expected published ports"
    assert all(p.startswith("127.0.0.1:") for p in ports), ports
    assert ":latest" not in text
    assert "FLASK_DEBUG" not in text
    assert "SECRET_KEY" not in text
    assert "mnemonic" not in text.lower().replace("public mnemonic", "")


def test_gitignore_covers_env_and_keys():
    ignored = _read(".gitignore")
    for pattern in (".env", "private_keys.json", "*.db"):
        assert pattern in ignored


def test_removed_insecure_scripts_stay_removed():
    for rel in ("scripts/create_admin.py", "scripts/create_lab_user.py",
                "scripts/fund_ganache_account.py", "scripts/use_standard_account.py",
                "scripts/setup_ganache.py", "app/routes/benchmark.py"):
        assert not os.path.exists(os.path.join(ROOT, rel)), rel
