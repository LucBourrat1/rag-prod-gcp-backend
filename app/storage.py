import asyncio
import os

from google.cloud import storage

BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "rag-prod-gcp-pdfs")
gcs_client = storage.Client()


async def upload_pdf_to_gcs(content: bytes, filename: str, user_email: str) -> str:
    safe_email = user_email.replace("@", "_").replace(".", "_")
    gcs_path = f"users/{safe_email}/{filename}"
    bucket = gcs_client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    # upload_from_string est synchrone — on le wrappe pour ne pas bloquer asyncio
    await asyncio.to_thread(
        blob.upload_from_string, content, content_type="application/pdf"
    )
    return gcs_path
