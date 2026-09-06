"""retain the RLS migration point for backward-compatible upgrades

Revision ID: 0008_remove_supabase_auth_rls
Revises: 0007_enable_rls_policies
"""

from __future__ import annotations

from alembic import op

revision = "0008_remove_supabase_auth_rls"
down_revision = "0007_enable_rls_policies"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
