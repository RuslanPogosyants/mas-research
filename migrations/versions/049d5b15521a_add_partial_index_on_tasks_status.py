"""add partial index on tasks status

Revision ID: 049d5b15521a
Revises: 0fe2e88121f2
Create Date: 2026-05-30 13:01:22.237719

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "049d5b15521a"
down_revision: str | Sequence[str] | None = "0fe2e88121f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "idx_tasks_status",
        "tasks",
        ["status"],
        postgresql_where=sa.text("status IN ('planning','running','partial_ready')"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_tasks_status", table_name="tasks")
