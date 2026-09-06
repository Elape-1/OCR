"""add document owner

Revision ID: 0005_add_document_owner
Revises: 0004_add_processing_timing
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0005_add_document_owner"
down_revision = "0004_add_processing_timing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("owner_id", sa.String(length=128), nullable=True))
    op.create_index("ix_documents_owner_id", "documents", ["owner_id"])


def downgrade() -> None:
    op.drop_index("ix_documents_owner_id", table_name="documents")
    op.drop_column("documents", "owner_id")