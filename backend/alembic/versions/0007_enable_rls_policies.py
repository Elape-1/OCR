"""enable owner-based row-level security

Revision ID: 0007_enable_rls_policies
Revises: 0006_add_page_storage_key
"""

from alembic import op

revision = "0007_enable_rls_policies"
down_revision = "0006_add_page_storage_key"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return

    op.execute("ALTER TABLE documents ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE pages ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE attributes ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE corrections ENABLE ROW LEVEL SECURITY")

    op.execute("CREATE POLICY documents_owner_policy ON documents FOR ALL TO authenticated USING (owner_id = (select auth.uid()::text)) WITH CHECK (owner_id = (select auth.uid()::text))")
    op.execute("CREATE POLICY pages_owner_policy ON pages FOR ALL TO authenticated USING (EXISTS (SELECT 1 FROM documents WHERE documents.id = pages.document_id AND documents.owner_id = (select auth.uid()::text))) WITH CHECK (EXISTS (SELECT 1 FROM documents WHERE documents.id = pages.document_id AND documents.owner_id = (select auth.uid()::text)))")
    op.execute("CREATE POLICY attributes_owner_policy ON attributes FOR ALL TO authenticated USING (EXISTS (SELECT 1 FROM documents WHERE documents.id = attributes.document_id AND documents.owner_id = (select auth.uid()::text))) WITH CHECK (EXISTS (SELECT 1 FROM documents WHERE documents.id = attributes.document_id AND documents.owner_id = (select auth.uid()::text)))")
    op.execute("CREATE POLICY corrections_owner_policy ON corrections FOR ALL TO authenticated USING (EXISTS (SELECT 1 FROM attributes JOIN documents ON documents.id = attributes.document_id WHERE attributes.id = corrections.attribute_id AND documents.owner_id = (select auth.uid()::text))) WITH CHECK (EXISTS (SELECT 1 FROM attributes JOIN documents ON documents.id = attributes.document_id WHERE attributes.id = corrections.attribute_id AND documents.owner_id = (select auth.uid()::text)))")


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return

    op.execute("DROP POLICY IF EXISTS corrections_owner_policy ON corrections")
    op.execute("DROP POLICY IF EXISTS attributes_owner_policy ON attributes")
    op.execute("DROP POLICY IF EXISTS pages_owner_policy ON pages")
    op.execute("DROP POLICY IF EXISTS documents_owner_policy ON documents")
    op.execute("ALTER TABLE corrections DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE attributes DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE pages DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE documents DISABLE ROW LEVEL SECURITY")
