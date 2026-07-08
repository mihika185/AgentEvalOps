import {
    useEffect,
    useState
} from "react";
import {
    AlertTriangle,
    FileText,
    Layers,
    Loader2,
    RefreshCw,
} from "lucide-react";

import {
    fetchJson
} from "./api";
import type {
    DocumentChunk,
    DocumentRecord
} from "./types";

export default function DocumentExplorer() {
    const [documents, setDocuments] = useState<DocumentRecord[]>([]);
    const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(
        null
    );
    const [chunks, setChunks] = useState<DocumentChunk[]>([]);
    const [loadingDocuments, setLoadingDocuments] = useState(true);
    const [loadingChunks, setLoadingChunks] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const selectedDocument =
        documents.find((document) => document.id === selectedDocumentId) || null;

    async function loadDocuments() {
        try {
            setLoadingDocuments(true);
            setError(null);

            const data = await fetchJson<DocumentRecord[]>("/documents?limit=50");
            setDocuments(data);

            if (data.length > 0 && !selectedDocumentId) {
                setSelectedDocumentId(data[0].id);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load documents");
        } finally {
            setLoadingDocuments(false);
        }
    }

    async function loadChunks(documentId: string) {
        try {
            setLoadingChunks(true);
            setError(null);

            const data = await fetchJson<DocumentChunk[]>(
                `/documents/${documentId}/chunks?limit=200`
            );

            setChunks(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load chunks");
            setChunks([]);
        } finally {
            setLoadingChunks(false);
        }
    }

    useEffect(() => {
        loadDocuments();
    }, []);

    useEffect(() => {
        if (selectedDocumentId) {
            loadChunks(selectedDocumentId);
        }
    }, [selectedDocumentId]);

    return (
        <section id="document-explorer" className="document-explorer">
            <div className="document-explorer-header">
                <div>
                    <p className="eyebrow">Document Explorer</p>
                    <h2>Indexed documents and chunks</h2>
                    <p>
                        Inspect the documents that power retrieval and review the exact
                        chunks available to the RAG pipeline.
                    </p>
                </div>

                <button className="ghost-button" type="button" onClick={loadDocuments}>
                    <RefreshCw size={17} />
                    Refresh
                </button>
            </div>

            {error ? (
                <div className="rag-error">
                    <AlertTriangle size={18} />
                    <span>{error}</span>
                </div>
            ) : null}

            <div className="document-layout">
                <aside className="document-list-panel">
                    <div className="mini-panel-title">
                        <FileText size={18} />
                        <h3>Documents ({documents.length})</h3>
                    </div>

                    {loadingDocuments ? (
                        <LoadingState label="Loading documents..." />
                    ) : documents.length === 0 ? (
                        <EmptyState label="No documents indexed yet." />
                    ) : (
                        <div className="document-list">
                            {documents.map((document) => (
                                <DocumentListItem
                                    key={document.id}
                                    document={document}
                                    selected={document.id === selectedDocumentId}
                                    onSelect={() => setSelectedDocumentId(document.id)}
                                />
                            ))}
                        </div>
                    )}
                </aside>

                <section className="chunk-panel">
                    <div className="chunk-panel-header">
                        <div className="mini-panel-title">
                            <Layers size={18} />
                            <h3>
                                Chunks{" "}
                                {selectedDocument ? `for ${selectedDocument.filename}` : ""}
                            </h3>
                        </div>

                        {selectedDocument ? (
                            <span className="document-id-pill">{selectedDocument.id}</span>
                        ) : null}
                    </div>

                    {selectedDocument ? (
                        <DocumentSummary document={selectedDocument} />
                    ) : null}

                    {loadingChunks ? (
                        <LoadingState label="Loading chunks..." />
                    ) : !selectedDocument ? (
                        <EmptyState label="Select a document to inspect chunks." />
                    ) : chunks.length === 0 ? (
                        <EmptyState label="No chunks found for this document." />
                    ) : (
                        <div className="chunk-list">
                            {chunks.map((chunk) => (
                                <ChunkCard key={chunk.id} chunk={chunk} />
                            ))}
                        </div>
                    )}
                </section>
            </div>
        </section>
    );
}

function DocumentListItem({
    document,
    selected,
    onSelect,
}: {
    document: DocumentRecord;
    selected: boolean;
    onSelect: () => void;
}) {
    return (
        <button
            className={selected ? "document-card selected-document" : "document-card"}
            onClick={onSelect}
            type="button"
        >
            <div>
                <strong>{document.filename}</strong>
                <p>{document.id}</p>
            </div>

            <div className="document-card-footer">
                <span className="badge neutral">{document.status}</span>
                <span>{formatNumber(document.num_chunks)} chunks</span>
            </div>
        </button>
    );
}

function DocumentSummary({ document }: { document: DocumentRecord }) {
    return (
        <div className="document-summary">
            <InfoBox label="File type" value={document.file_type} />
            <InfoBox label="Status" value={document.status} />
            <InfoBox label="Pages" value={formatNumber(document.num_pages)} />
            <InfoBox label="Chunks" value={formatNumber(document.num_chunks)} />
            <InfoBox label="Created" value={formatDate(document.created_at)} />
            <InfoBox label="Updated" value={formatDate(document.updated_at)} />
        </div>
    );
}

function InfoBox({ label, value }: { label: string; value: string }) {
    return (
        <div className="info-box">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
    );
}

function ChunkCard({ chunk }: { chunk: DocumentChunk }) {
    const text = chunk.chunk_text || "No text available for this chunk.";
    const metadata = chunk.metadata_json || {};

    return (
        <article className="chunk-card">
            <div className="chunk-card-top">
                <div>
                    <strong>Chunk #{chunk.chunk_index}</strong>
                    <p>{chunk.id}</p>
                </div>

                <span className="chunk-index">index {chunk.chunk_index}</span>
            </div>

            <p className="chunk-text">{text}</p>

            <div className="chunk-meta-row">
                <span>{chunk.document_id}</span>
                {typeof metadata.start_char === "number" ? (
                    <span>start {metadata.start_char}</span>
                ) : null}
                {typeof metadata.end_char === "number" ? (
                    <span>end {metadata.end_char}</span>
                ) : null}
                {chunk.token_count !== undefined && chunk.token_count !== null ? (
                    <span>{chunk.token_count} tokens</span>
                ) : null}
            </div>

            {Object.keys(metadata).length > 0 ? (
                <details className="metadata-details">
                    <summary>Metadata</summary>
                    <pre>{JSON.stringify(metadata, null, 2)}</pre>
                </details>
            ) : null}
        </article>
    );
}

function LoadingState({ label }: { label: string }) {
    return (
        <div className="document-state">
            <Loader2 className="spin" size={22} />
            <span>{label}</span>
        </div>
    );
}

function EmptyState({ label }: { label: string }) {
    return <div className="document-state muted-text">{label}</div>;
}

function formatNumber(value: number | null | undefined) {
    if (value === null || value === undefined) {
        return "—";
    }

    return new Intl.NumberFormat("en-US").format(Number(value));
}

function formatDate(value: string | null | undefined) {
    if (!value) {
        return "—";
    }

    return new Date(value).toLocaleDateString();
}