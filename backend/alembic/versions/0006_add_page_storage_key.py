"""add durable storage key for page images

Revision ID: 0006_add_page_storage_key
Revises: 0005_add_document_owner
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0006_add_page_storage_key"
down_revision = "0005_add_document_owner"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("pages", sa.Column("storage_key", sa.String(length=512), nullable=True))


def downgrade() -> None:
    op.drop_column("pages", "storage_key")