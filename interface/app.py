"""Streamlit UI for the banking signature verification system.

Run with:
    streamlit run interface/app.py
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Allow running directly from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from config.settings import Settings
from container import build_services

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Signature Verification System",
    page_icon="🏦",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state: initialize services once
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model and services…")
def get_services():
    settings = Settings()
    try:
        enrollment, verification, identification, audit = build_services(settings)
        return enrollment, verification, identification, audit, settings, None
    except RuntimeError as exc:
        return None, None, None, None, settings, str(exc)


enrollment_svc, verification_svc, identification_svc, audit_logger, settings, init_error = get_services()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – navigation
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🏦 Signature Verification")
st.sidebar.caption("Banking-grade handwritten signature verification")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📋 Enroll Client", "🔍 Verify Signature", "🔎 Identify Signature", "📊 Database Stats"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Error banner if model not loaded
# ─────────────────────────────────────────────────────────────────────────────

if init_error:
    st.error(f"⚠️ System initialization failed: {init_error}")
    st.info("Run `python siamese_encoder.py` to train the model, then restart the app.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file) -> str | None:
    """Save an uploaded file to a temp location and return the path."""
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

# ── Home ──────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🏦 Handwritten Signature Verification System")
    st.markdown(
        """
        This system uses a **Siamese CNN** trained with triplet loss to verify and
        identify handwritten signatures in a banking environment.

        ---

        ### Available Operations

        | Operation | Description |
        |-----------|-------------|
        | **Enroll** | Register a new client using 1–N reference signatures |
        | **Verify** | 1:1 – check if a signature belongs to a specific client |
        | **Identify** | 1:N – find the best matching client for an unknown signature |

        ---

        ### How it works

        1. Reference signatures are encoded into **128-dimensional L2-normalized embeddings**.
        2. At verification/identification time, the query signature is encoded and compared
           using **cosine distance** against stored embeddings.
        3. A decision is made based on a configurable **distance threshold** (default: `{threshold}`).
        """.format(threshold=settings.verification_threshold)
    )

    from storage.embedding_repository import EmbeddingRepository
    repo = EmbeddingRepository(db_path=settings.embeddings_db_path)
    st.metric("Enrolled Clients", repo.count())

# ── Enroll ────────────────────────────────────────────────────────────────────
elif page == "📋 Enroll Client":
    st.title("📋 Enroll New Client")

    col1, col2 = st.columns([1, 2])

    with col1:
        client_id = st.text_input("Client ID", placeholder="e.g. 00042")
        overwrite = st.checkbox("Overwrite if already enrolled", value=False)
        notes = st.text_area("Notes (optional)", height=80)

    with col2:
        uploaded_files = st.file_uploader(
            "Upload reference signatures",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg", "bmp", "tif"],
            help=f"Minimum {settings.min_signatures_required} signature(s) required. "
                 f"Recommended: {settings.recommended_signatures}+.",
        )

    if st.button("Enroll Client", type="primary"):
        if not client_id:
            st.error("Please enter a Client ID.")
        elif not uploaded_files:
            st.error("Please upload at least one signature image.")
        else:
            # Save files to temp
            tmp_paths = [save_uploaded_file(f) for f in uploaded_files if f]
            tmp_paths = [p for p in tmp_paths if p]

            with st.spinner("Encoding signatures…"):
                result = enrollment_svc.enroll(
                    client_id=client_id.strip(),
                    signature_paths=tmp_paths,
                    overwrite=overwrite,
                    notes=notes,
                )

            # Clean up temp files
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

            if result.success:
                action = "updated" if result.is_update else "enrolled"
                st.success(
                    f"✅ Client **{result.client_id}** {action} successfully!\n\n"
                    f"- **Signatures enrolled:** {result.num_enrolled}\n"
                    f"- **Skipped (unreadable):** {result.num_skipped}"
                )
            else:
                st.error(f"❌ Enrollment failed: {result.message}")

# ── Verify ────────────────────────────────────────────────────────────────────
elif page == "🔍 Verify Signature":
    st.title("🔍 1:1 Signature Verification")
    st.caption("Check whether a signature belongs to a specific client.")

    col1, col2 = st.columns([1, 2])

    with col1:
        client_id = st.text_input("Client ID to verify against", placeholder="e.g. 00042")
        threshold = st.slider(
            "Distance threshold",
            min_value=0.05,
            max_value=0.50,
            value=settings.verification_threshold,
            step=0.01,
            help="Cosine distance below this value → AUTHENTIC",
        )

    with col2:
        uploaded = st.file_uploader(
            "Upload query signature",
            type=["png", "jpg", "jpeg", "bmp", "tif"],
            help="The signature to verify.",
        )
        if uploaded:
            st.image(uploaded, caption="Query signature", use_column_width=True)

    if st.button("Verify", type="primary"):
        if not client_id:
            st.error("Please enter a Client ID.")
        elif not uploaded:
            st.error("Please upload a signature image.")
        else:
            tmp_path = save_uploaded_file(uploaded)
            if tmp_path:
                with st.spinner("Verifying…"):
                    result = verification_svc.verify(
                        client_id=client_id.strip(),
                        signature_path=tmp_path,
                        threshold=threshold,
                    )
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

                if result.status == "ERROR":
                    st.error(f"❌ Error: {result.message}")
                elif result.is_authentic:
                    st.success(
                        f"✅ **AUTHENTIC** – Signature matches client **{result.client_id}**\n\n"
                        f"- **Cosine distance:** {result.cosine_distance:.4f}\n"
                        f"- **Threshold:** {result.threshold_used}\n"
                        f"- **Confidence:** {result.confidence_pct:.1f}%"
                    )
                else:
                    st.error(
                        f"🚨 **REJECTED** – Signature does NOT match client **{result.client_id}**\n\n"
                        f"- **Cosine distance:** {result.cosine_distance:.4f}\n"
                        f"- **Threshold:** {result.threshold_used}\n"
                        f"- **Confidence:** {result.confidence_pct:.1f}%"
                    )

# ── Identify ──────────────────────────────────────────────────────────────────
elif page == "🔎 Identify Signature":
    st.title("🔎 1:N Signature Identification")
    st.caption("Find the best matching client for an unknown signature.")

    col1, col2 = st.columns([1, 2])

    with col1:
        threshold = st.slider(
            "Distance threshold",
            min_value=0.05,
            max_value=0.50,
            value=settings.identification_threshold,
            step=0.01,
            help="Cosine distance below this value → IDENTIFIED",
        )
        top_k = st.number_input("Top K results", min_value=1, max_value=20, value=settings.identification_top_k)

    with col2:
        uploaded = st.file_uploader(
            "Upload query signature",
            type=["png", "jpg", "jpeg", "bmp", "tif"],
        )
        if uploaded:
            st.image(uploaded, caption="Query signature", use_column_width=True)

    if st.button("Identify", type="primary"):
        if not uploaded:
            st.error("Please upload a signature image.")
        else:
            tmp_path = save_uploaded_file(uploaded)
            if tmp_path:
                with st.spinner("Searching all enrolled clients…"):
                    result = identification_svc.identify(
                        signature_path=tmp_path,
                        top_k=int(top_k),
                        threshold=threshold,
                    )
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

                if result.status == "ERROR":
                    st.error(f"❌ Error: {result.message}")
                else:
                    status_map = {
                        "IDENTIFIED": ("✅", "success"),
                        "UNCERTAIN": ("⚠️", "warning"),
                        "UNKNOWN": ("🚨", "error"),
                    }
                    icon, color = status_map.get(result.status, ("ℹ️", "info"))

                    if result.status == "IDENTIFIED":
                        st.success(
                            f"{icon} **{result.status}** – Best match: "
                            f"client **{result.best_match.client_id}**\n\n"
                            f"- **Cosine distance:** {result.best_match.cosine_distance:.4f}\n"
                            f"- **Confidence:** {result.best_match.confidence_pct:.1f}%"
                        )
                    elif result.status == "UNCERTAIN":
                        st.warning(
                            f"{icon} **{result.status}** – Possible match: "
                            f"client **{result.best_match.client_id}** (low confidence)\n\n"
                            f"- **Cosine distance:** {result.best_match.cosine_distance:.4f}\n"
                            f"- **Confidence:** {result.best_match.confidence_pct:.1f}%"
                        )
                    else:
                        st.error(
                            f"{icon} **{result.status}** – No matching client found.\n\n"
                            f"Closest: client **{result.best_match.client_id}** "
                            f"(dist={result.best_match.cosine_distance:.4f})"
                        )

                    if result.top_candidates:
                        st.subheader("Top Candidates")
                        import pandas as pd
                        rows = [
                            {
                                "Rank": i + 1,
                                "Client ID": c.client_id,
                                "Cosine Distance": f"{c.cosine_distance:.4f}",
                                "Confidence %": f"{c.confidence_pct:.1f}%",
                            }
                            for i, c in enumerate(result.top_candidates)
                        ]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Database Stats ─────────────────────────────────────────────────────────────
elif page == "📊 Database Stats":
    st.title("📊 Embedding Database Statistics")

    from storage.client_repository import ClientRepository
    from storage.embedding_repository import EmbeddingRepository

    client_repo = ClientRepository()
    embedding_repo = EmbeddingRepository(db_path=settings.embeddings_db_path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Enrolled Clients", embedding_repo.count())
    total_sigs = sum(
        len(embedding_repo.get(cid))
        for cid in embedding_repo.all_clients()
        if embedding_repo.get(cid) is not None
    )
    col2.metric("Total Embeddings", total_sigs)
    avg = total_sigs / embedding_repo.count() if embedding_repo.count() > 0 else 0
    col3.metric("Avg Embeddings / Client", f"{avg:.1f}")

    st.subheader("Enrolled Clients")

    import pandas as pd

    rows = []
    for cid in embedding_repo.all_clients():
        embs = embedding_repo.get(cid)
        meta = client_repo.get(cid)
        rows.append(
            {
                "Client ID": cid,
                "Embeddings": len(embs) if embs is not None else 0,
                "Enrolled At": meta.enrolled_at if meta else "—",
                "Updated At": meta.updated_at if meta else "—",
                "Notes": meta.notes if meta else "—",
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No clients enrolled yet.")

    st.subheader("Configuration")
    cfg_data = {
        "Model path": settings.model_weights_path,
        "Embeddings DB": settings.embeddings_db_path,
        "Verification threshold": settings.verification_threshold,
        "Identification threshold": settings.identification_threshold,
        "Embedding size": settings.embedding_size,
        "Input image size": str(settings.input_image_size),
        "Audit log": settings.audit_log_path,
    }
    st.json(cfg_data)




