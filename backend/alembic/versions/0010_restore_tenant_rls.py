"""restore tenant isolation for direct backend sessions

Revision ID: 0010_restore_tenant_rls
Revises: 0009_attribute_page_document_fk
"""

from __future__ import annotations

from alembic import op


revision = "0010_restore_tenant_rls"
down_revision = "0009_attribute_page_document_fk"
branch_labels = None
depends_on = None


TABLES = ("corrections", "attributes", "pages", "documents")
POLICIES = (
    ("corrections_owner_policy", "corrections"),
    ("attributes_owner_policy", "attributes"),
    ("pages_owner_policy", "pages"),
    ("documents_owner_policy", "documents"),
)


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return

    for policy_name, table_name in POLICIES:
        op.execute(f"DROP POLICY IF EXISTS {policy_name} ON {table_name}")
    for table_name in TABLES:
        op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")

    owner_context = "nullif(current_setting('app.user_id', true), '')"
    op.execute(
        f"CREATE POLICY documents_owner_policy ON documents FOR ALL "
        f"USING (owner_id = {owner_context}) WITH CHECK (owner_id = {owner_context})"
    )
    op.execute(
        f"CREATE POLICY pages_owner_policy ON pages FOR ALL "
        f"USING (EXISTS (SELECT 1 FROM documents WHERE documents.id = pages.document_id AND documents.owner_id = {owner_context})) "
        f"WITH CHECK (EXISTS (SELECT 1 FROM documents WHERE documents.id = pages.document_id AND documents.owner_id = {owner_context}))"
    )
    op.execute(
        f"CREATE POLICY attributes_owner_policy ON attributes FOR ALL "
        f"USING (EXISTS (SELECT 1 FROM documents WHERE documents.id = attributes.document_id AND documents.owner_id = {owner_context})) "
        f"WITH CHECK (EXISTS (SELECT 1 FROM documents WHERE documents.id = attributes.document_id AND documents.owner_id = {owner_context}))"
    )
    op.execute(
        f"CREATE POLICY corrections_owner_policy ON corrections FOR ALL "
        f"USING (EXISTS (SELECT 1 FROM attributes JOIN documents ON documents.id = attributes.document_id "
        f"WHERE attributes.id = corrections.attribute_id AND documents.owner_id = {owner_context})) "
        f"WITH CHECK (EXISTS (SELECT 1 FROM attributes JOIN documents ON documents.id = attributes.document_id "
        f"WHERE attributes.id = corrections.attribute_id AND documents.owner_id = {owner_context}))"
    )


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return

    for policy_name, table_name in POLICIES:
        op.execute(f"DROP POLICY IF EXISTS {policy_name} ON {table_name}")
    for table_name in TABLES:
        op.execute(f"ALTER TABLE {table_name} DISABLE ROW LEVEL SECURITY")