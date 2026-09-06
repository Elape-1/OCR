"""add document processing timing fields

Revision ID: 0004_add_processing_timing
Revises: 0003_stage_attributes_and_source
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0004_add_processing_timing"
down_revision = "0003_stage_attributes_and_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("processing_finished_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("processing_duration_seconds", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "processing_duration_seconds")
    op.drop_column("documents", "processing_finished_at")
    op.drop_column("documents", "processing_started_at")