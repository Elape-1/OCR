"""store page-aware spans for multi-page attributes

Revision ID: 0011_add_attribute_page_spans
Revises: 0010_restore_tenant_rls
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0011_add_attribute_page_spans"
down_revision = "0010_restore_tenant_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("attributes", recreate="always") as batch_op:
            batch_op.add_column(sa.Column("page_spans", sa.JSON(), nullable=False, server_default="[]"))
        return

    op.add_column("attributes", sa.Column("page_spans", sa.JSON(), nullable=False, server_default=sa.text("'[]'")))


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("attributes", recreate="always") as batch_op:
            batch_op.drop_column("page_spans")
        return

    op.drop_column("attributes", "page_spans")