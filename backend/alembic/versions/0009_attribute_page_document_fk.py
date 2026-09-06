"""enforce that attributes reference a page in the same document

Revision ID: 0009_attribute_page_document_fk
Revises: 0008_remove_supabase_auth_rls
"""

from __future__ import annotations

from alembic import op


revision = "0009_attribute_page_document_fk"
down_revision = "0008_remove_supabase_auth_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("pages", recreate="always") as batch_op:
            batch_op.create_unique_constraint("uq_pages_document_id_id", ["document_id", "id"])
        with op.batch_alter_table("attributes", recreate="always") as batch_op:
            batch_op.create_foreign_key(
                "fk_attributes_page_document",
                "pages",
                ["document_id", "page_id"],
                ["document_id", "id"],
                ondelete="CASCADE",
            )
        return

    op.create_unique_constraint("uq_pages_document_id_id", "pages", ["document_id", "id"])
    op.create_foreign_key(
        "fk_attributes_page_document",
        "attributes",
        "pages",
        ["document_id", "page_id"],
        ["document_id", "id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("attributes", recreate="always") as batch_op:
            batch_op.drop_constraint("fk_attributes_page_document", type_="foreignkey")
        with op.batch_alter_table("pages", recreate="always") as batch_op:
            batch_op.drop_constraint("uq_pages_document_id_id", type_="unique")
        return

    op.drop_constraint("fk_attributes_page_document", "attributes", type_="foreignkey")
    op.drop_constraint("uq_pages_document_id_id", "pages", type_="unique")