import{ 
  useState 
} from "react";
import type{ 
  FormEvent, ReactNode 
} from "react";
import {
  AlertTriangle,
  FileText,
  Loader2,
  PlayCircle,
  Quote,
  Search,
  ShieldCheck,
} from "lucide-react";

import{ 
  postJson 
} from "./api";
import type {
  RAGAnswerRequest,
  RAGAnswerResponse,
  RAGCitation,
  RAGEvaluationMetric,
  RAGSourceChunk,
  RetrievalProvider,
} from "./types";

type RagPlaygroundProps = {
  onInspectRun: (runId: string) => void;
};

export default function RagPlayground({ onInspectRun }: RagPlaygroundProps) {
  const [query, setQuery] = useState(
    "What must customers provide when reporting a damaged product?"
  );
  const [retrievalProvider, setRetrievalProvider] =
    useState<RetrievalProvider>("hybrid");
  const [topK, setTopK] = useState(3);
  const [rerank, setRerank] = useState(true);
  const [candidateMultiplier, setCandidateMultiplier] = useState(3);
  const [qualityGateProfile, setQualityGateProfile] = useState("default-v1");
  const [documentId, setDocumentId] = useState("");
  const [experimentId, setExperimentId] = useState("");

  const [result, setResult] = useState<RAGAnswerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  async function runPlayground(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!query.trim()) {
      setError("Query cannot be empty.");
      return;
    }

    const payload: RAGAnswerRequest = {
      query: query.trim(),
      top_k: topK,
      rerank,
      candidate_multiplier: candidateMultiplier,
      retrieval_provider: retrievalProvider,
      quality_gate_profile: qualityGateProfile.trim() || "default-v1",
    };

    if (documentId.trim()) {
      payload.document_id = documentId.trim();
    }

    if (experimentId.trim()) {
      payload.experiment_id = experimentId.trim();
    }

    try {
      setRunning(true);
      setError(null);

      const data = await postJson<RAGAnswerResponse, RAGAnswerRequest>(
        "/rag/answer",
        payload
      );

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run RAG answer");
    } finally {
      setRunning(false);
    }
  }

  return (
    <section id="rag-playground" className="rag-playground">
      <div className="rag-playground-header">
        <div>
          <p className="eyebrow">RAG Playground</p>
          <h2>Ask the indexed knowledge base</h2>
          <p>
            Run the RAG workflow from the UI and inspect answer quality,
            citations, source chunks, latency, tokens, and gates.
          </p>
        </div>

        {result ? (
          <button
            className="ghost-button"
            type="button"
            onClick={() => onInspectRun(result.run_id)}
          >
            <Search size={17} />
            Inspect Run
          </button>
        ) : null}
      </div>

      <form className="rag-form" onSubmit={runPlayground}>
        <label className="query-box">
          <span>Question</span>
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={3}
            placeholder="Ask a question about the indexed documents..."
          />
        </label>

        <div className="rag-controls">
          <ControlGroup label="Retriever">
            <select
              value={retrievalProvider}
              onChange={(event) =>
                setRetrievalProvider(event.target.value as RetrievalProvider)
              }
            >
              <option value="dense">Dense</option>
              <option value="bm25">BM25</option>
              <option value="hybrid">Hybrid</option>
            </select>
          </ControlGroup>

          <ControlGroup label="Top K">
            <input
              min={1}
              max={20}
              type="number"
              value={topK}
              onChange={(event) => setTopK(safeNumber(event.target.value, 3))}
            />
          </ControlGroup>

          <ControlGroup label="Candidate Multiplier">
            <input
              min={1}
              max={10}
              type="number"
              value={candidateMultiplier}
              onChange={(event) =>
                setCandidateMultiplier(safeNumber(event.target.value, 3))
              }
            />
          </ControlGroup>

          <ControlGroup label="Quality Gate Profile">
            <input
              value={qualityGateProfile}
              onChange={(event) => setQualityGateProfile(event.target.value)}
              placeholder="default-v1"
            />
          </ControlGroup>

          <ControlGroup label="Document ID optional">
            <input
              value={documentId}
              onChange={(event) => setDocumentId(event.target.value)}
              placeholder="doc_..."
            />
          </ControlGroup>

          <ControlGroup label="Experiment ID optional">
            <input
              value={experimentId}
              onChange={(event) => setExperimentId(event.target.value)}
              placeholder="exp_..."
            />
          </ControlGroup>

          <label className="checkbox-control">
            <input
              type="checkbox"
              checked={rerank}
              onChange={(event) => setRerank(event.target.checked)}
            />
            <span>Use reranker</span>
          </label>

          <button className="run-rag-button" type="submit" disabled={running}>
            {running ? (
              <>
                <Loader2 className="spin" size={18} />
                Running...
              </>
            ) : (
              <>
                <PlayCircle size={18} />
                Run RAG
              </>
            )}
          </button>
        </div>
      </form>

      {error ? (
        <div className="rag-error">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      ) : null}

      {result ? <RagResult result={result} onInspectRun={onInspectRun} /> : null}
    </section>
  );
}

function RagResult({
  result,
  onInspectRun,
}: {
  result: RAGAnswerResponse;
  onInspectRun: (runId: string) => void;
}) {
  return (
    <div className="rag-result">
      <div className="rag-result-top">
        <div>
          <p className="eyebrow">Answer Result</p>
          <h3>{result.run_id}</h3>
        </div>

        <div className="rag-badge-row">
          <span className={result.quality_gate_passed ? "badge success" : "badge danger"}>
            {result.quality_gate_passed ? "gates passed" : "gates failed"}
          </span>
          <span className={result.citation_check_passed ? "badge success" : "badge danger"}>
            {result.citation_check_passed ? "citations passed" : "citations failed"}
          </span>
          <button
            className="ghost-button compact-button"
            type="button"
            onClick={() => onInspectRun(result.run_id)}
          >
            Inspect
          </button>
        </div>
      </div>

      <div className="rag-answer-card">
        <h4>Answer</h4>
        <p>{result.answer}</p>
      </div>

      <div className="rag-summary-grid">
        <MetricBox label="Latency" value={`${result.total_latency_ms} ms`} />
        <MetricBox label="Total Tokens" value={formatNumber(result.total_tokens)} />
        <MetricBox label="Estimated Cost" value={`$${result.estimated_cost.toFixed(4)}`} />
        <MetricBox label="Retrieved Chunks" value={formatNumber(result.retrieved_chunk_count)} />
        <MetricBox label="Citation Accuracy" value={formatScore(result.citation_accuracy_score)} />
        <MetricBox label="Gate Pass Rate" value={formatPercent(result.quality_gate_pass_rate)} />
      </div>

      {result.failed_quality_gates.length > 0 ? (
        <div className="rag-warning">
          <AlertTriangle size={18} />
          <div>
            <strong>Failed quality gates</strong>
            <p>{result.failed_quality_gates.join(", ")}</p>
          </div>
        </div>
      ) : null}

      {result.citation_failed_reasons.length > 0 ? (
        <div className="rag-warning">
          <AlertTriangle size={18} />
          <div>
            <strong>Citation issues</strong>
            <p>{result.citation_failed_reasons.join(", ")}</p>
          </div>
        </div>
      ) : null}

      <div className="rag-two-column">
        <ResultSection
          title={`Citations (${result.citations.length})`}
          icon={<Quote size={18} />}
        >
          <div className="rag-card-list">
            {result.citations.length === 0 ? (
              <p className="muted-text">No citations returned.</p>
            ) : (
              result.citations.map((citation) => (
                <CitationCard key={citation.source_number} citation={citation} />
              ))
            )}
          </div>
        </ResultSection>

        <ResultSection
          title={`Evaluation Metrics (${result.evaluation_metrics.length})`}
          icon={<ShieldCheck size={18} />}
        >
          <div className="rag-metric-grid">
            {result.evaluation_metrics.map((metric) => (
              <EvaluationMetricCard
                key={metric.metric_name}
                metric={metric}
              />
            ))}
          </div>
        </ResultSection>
      </div>

      <ResultSection
        title={`Source Chunks (${result.source_chunks.length})`}
        icon={<FileText size={18} />}
      >
        <div className="rag-card-list">
          {result.source_chunks.map((chunk) => (
            <SourceChunkCard key={chunk.chunk_id} chunk={chunk} />
          ))}
        </div>
      </ResultSection>
    </div>
  );
}

function ControlGroup({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="control-group">
      <span>{label}</span>
      {children}
    </label>
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rag-metric-box">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ResultSection({
  title,
  icon,
  children,
}: {
  title: string;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="rag-result-section">
      <div className="detail-title">
        {icon}
        <h3>{title}</h3>
      </div>
      {children}
    </div>
  );
}

function CitationCard({ citation }: { citation: RAGCitation }) {
  return (
    <div className="rag-item-card">
      <div className="rag-item-top">
        <strong>Source {citation.source_number}</strong>
        <span>{formatScore(citation.support_score)} support</span>
      </div>
      <p>{citation.text_excerpt}</p>
      <div className="tag-row">
        <span>{citation.chunk_id}</span>
        <span>{citation.document_id}</span>
        <span>{formatScore(citation.retrieval_score)} retrieval</span>
      </div>
      {citation.matched_terms.length > 0 ? (
        <div className="matched-terms">
          {citation.matched_terms.map((term) => (
            <span key={term}>{term}</span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function SourceChunkCard({ chunk }: { chunk: RAGSourceChunk }) {
  return (
    <div className="rag-item-card">
      <div className="rag-item-top">
        <strong>{chunk.chunk_id}</strong>
        <span>{formatScore(chunk.score)} score</span>
      </div>
      <p>{chunk.text}</p>
      <div className="tag-row">
        <span>{chunk.document_id}</span>
      </div>
    </div>
  );
}

function EvaluationMetricCard({
  metric,
}: {
  metric: RAGEvaluationMetric;
}) {
  return (
    <div className="metric-result-card">
      <p>{toLabel(metric.metric_name)}</p>
      <strong>{formatScore(metric.metric_value)}</strong>
      <span>rag evaluator</span>
    </div>
  );
}

function safeNumber(value: string, fallback: number) {
  const parsed = Number(value);

  if (Number.isNaN(parsed)) {
    return fallback;
  }

  return parsed;
}

function toLabel(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatScore(value: number) {
  if (Number.isInteger(value)) {
    return value.toString();
  }

  return Number(value).toFixed(4);
}

function formatPercent(value: number) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}