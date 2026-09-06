"""add document type to documents

Revision ID: 0002_add_document_type
Revises: 0001_initial_schema
Create Date: 2026-07-22 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_add_document_type"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("document_type", sa.String(length=64), nullable=False, server_default=sa.text("'unknown'")),
    )


def downgrade() -> None:
    op.drop_column("documents", "document_type")