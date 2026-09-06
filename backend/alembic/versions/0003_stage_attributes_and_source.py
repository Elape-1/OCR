"""stage extracted attributes and add attribute source

Revision ID: 0003_stage_attributes_and_source
Revises: 0002_add_document_type
Create Date: 2026-08-07 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_stage_attributes_and_source"
down_revision = "0002_add_document_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "extracted_attributes",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    op.add_column(
        "attributes",
        sa.Column(
            "source",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'model'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("attributes", "source")
    op.drop_column("documents", "extracted_attributes")
