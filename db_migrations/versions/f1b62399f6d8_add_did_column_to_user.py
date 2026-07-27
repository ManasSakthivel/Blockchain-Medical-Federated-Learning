"""add did column to user

Revision ID: f1b62399f6d8
Revises: 
Create Date: 2026-07-27 22:03:50.382344

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f1b62399f6d8'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # SQLite does not support ALTER TABLE … ADD CONSTRAINT in the standard way.
    # We use raw DDL: ADD COLUMN (always works) + a separate CREATE UNIQUE INDEX.
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = [c['name'] for c in inspector.get_columns('user')]

    if 'did' not in existing_cols:
        op.execute(sa.text(
            "ALTER TABLE user ADD COLUMN did VARCHAR(120)"
        ))

    existing_indexes = [i['name'] for i in inspector.get_indexes('user')]
    if 'uq_user_did' not in existing_indexes:
        op.execute(sa.text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_did ON \"user\" (did)"
        ))


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_indexes = [i['name'] for i in inspector.get_indexes('user')]
    if 'uq_user_did' in existing_indexes:
        op.execute(sa.text("DROP INDEX IF EXISTS uq_user_did"))
    # SQLite cannot DROP COLUMN before version 3.35; we leave the column in place
    # on downgrade for broad compatibility. The column simply becomes unused.
