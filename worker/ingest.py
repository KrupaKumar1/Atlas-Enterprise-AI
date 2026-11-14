"""Ingestion worker: turns an uploaded file in S3 into searchable vectors in Pinecone.

Pipeline for one document:
  1. Claim the document row in Postgres (pending/failed -> processing).
  2. Download the file from S3 and validate its size and type.
  3. Extract text page by page (PDF, DOCX, or plain text).
  4. Split the text into overlapping chunks.
  5. Embed each chunk and upsert the vectors into the tenant's Pinecone namespace.
  6. Mark the document 'ready' and write an audit event (or 'failed' on any error).
"""
import io
import logging
from pathlib import Path
from docx import Document
from pypdf import PdfReader
from app.aws import client, embed
from app.vector import upsert
from app.config import get_settings as settings
from app.db import connection, audit
from app.observability import trace, initialize

# Set up Rius tracing (and load .env) once, before any traced function runs.
# Does nothing if RIUS_API_KEY is not set.
initialize()

# Log with timestamps under the 'atlas.worker' name so worker output is easy to tell apart.
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
log=logging.getLogger('atlas.worker')


def parse(blob:bytes, suffix:str):
    """Extract text from the raw file bytes.

    Returns a list of (page_number, text) tuples so each chunk can later be
    cited back to the page it came from.
    """
    # PDF: one entry per page. extract_text() can return None for image-only
    # pages, so fall back to an empty string.
    if suffix=='.pdf':
        reader=PdfReader(io.BytesIO(blob))
        return [(i+1,page.extract_text() or '') for i,page in enumerate(reader.pages)]
    # DOCX: Word files have no fixed pages, so join all paragraphs and treat
    # the whole document as page 1.
    if suffix=='.docx':
        doc=Document(io.BytesIO(blob))
        return [(1,'\n'.join(p.text for p in doc.paragraphs))]
    # Anything else (.txt, .md, ...) is treated as UTF-8 text on a single page.
    return [(1,blob.decode('utf-8'))]

def chunks(text, size=1200, overlap=180):
    """Yield fixed-size character windows over the text.

    Consecutive chunks share `overlap` characters so a sentence cut at a chunk
    boundary still appears whole in at least one chunk.
    """
    # Collapse all whitespace (newlines, tabs, repeated spaces) into single spaces.
    text=' '.join(text.split())
    pos=0
    while pos<len(text):
        yield text[pos:pos+size]
        # Stop once this window reached the end of the text, so we don't emit
        # a tiny trailing chunk that is fully contained in the previous one.
        if pos+size>=len(text): break
        # Move forward by (size - overlap) so the next chunk starts inside this one.
        pos+=size-overlap


@trace("process_s3_key")
def process_s3_key(key):
    """Ingest the document stored at S3 object `key` end to end."""
    s=settings()

    # --- Step 1: claim the document ---
    # SELECT ... FOR UPDATE locks the row until this transaction commits, so two
    # workers that receive the same key cannot both claim it.
    with connection() as conn:
        doc=conn.execute('SELECT * FROM documents WHERE s3_key=%s FOR UPDATE',(key,)).fetchone()
        # Unknown key, or already ingested: nothing to do.
        if not doc or doc['status']=='ready': return
        # Only new ('pending') or previously failed documents are picked up;
        # 'processing' means another worker is already on it.
        if doc['status'] not in ('pending','failed'): return
        conn.execute("UPDATE documents SET status='processing' WHERE id=%s",(doc['id'],))
    # Leaving the `with` block commits the 'processing' status and releases the lock.

    try:
        # --- Step 2: download and validate the file ---
        obj=client('s3').get_object(Bucket=s.s3_bucket,Key=key)
        # Reject oversized files before reading the body into memory.
        if obj['ContentLength']>s.max_upload_bytes: raise ValueError('Upload too large')
        # Read at most one byte past the limit; if we get that extra byte, the
        # file is too large even if ContentLength was wrong.
        blob=obj['Body'].read(s.max_upload_bytes+1)
        if len(blob)>s.max_upload_bytes: raise ValueError('Upload too large')
        # Every real PDF starts with the '%PDF' signature; reject files that
        # only have a .pdf extension.
        if Path(key).suffix.lower()=='.pdf' and not blob.startswith(b'%PDF'): raise ValueError('Invalid PDF')

        # --- Steps 3 and 4: extract text and split it into chunks ---
        pages=parse(blob,Path(key).suffix.lower())
        # Flatten into a single list of (page_number, chunk_text), keeping the
        # page number with every chunk for citations.
        parts=[(page,part) for page,text in pages for part in chunks(text)]
        # No text usually means a scanned PDF (images only), which needs OCR.
        if not parts: raise ValueError('No extractable text; scanned PDFs need Textract OCR')

        # --- Step 5: embed each chunk and build Pinecone records ---
        vectors=[]
        for i,(page,part) in enumerate(parts):
            vectors.append({
                # ID is '<document id>:<version>:<chunk index>', so re-ingesting
                # the same version overwrites its old vectors instead of duplicating them.
                'id':f'{doc["id"]}:{doc["current_version"]}:{i}',
                # Embedding vector for this chunk (currently from the Ollama embedding model).
                'values':embed(part),
                'metadata':{
                    # Returned with search hits so answers can cite title, page and text.
                    'document_id':str(doc['id']),
                    'title':doc['title'],'page':page,'text':part,
                    # Access control: search() filters on these so users only
                    # retrieve chunks they own or that are shared with their groups.
                    'owner_sub':doc['owner_sub'],'allowed_groups':doc['allowed_groups']}})
        # Keep batches small; Pinecone metadata must fit the index's limits.
        # Upsert 50 records at a time into the tenant's own namespace.
        for i in range(0,len(vectors),50): upsert(doc['tenant_id'],vectors[i:i+50])

        # --- Step 6: mark ready and record an audit event (one transaction) ---
        with connection() as conn:
            conn.execute("UPDATE documents SET status='ready' WHERE id=%s",(doc['id'],))
            audit(conn,doc['owner_sub'],doc['tenant_id'],'ingest_ready',{'document_id':str(doc['id']),'chunks':len(parts)})
        log.info('Ingested document %s chunks=%d',doc['id'],len(parts))
    except Exception:
        # Any failure (download, validation, parsing, embedding, Pinecone, DB):
        # log the full traceback, mark the document 'failed' so it can be retried
        # later, then re-raise so the caller/queue also sees the error.
        log.exception('Ingestion failed for key %s',key)
        with connection() as conn:
            conn.execute("UPDATE documents SET status='failed' WHERE id=%s",(doc['id'],))
        raise
