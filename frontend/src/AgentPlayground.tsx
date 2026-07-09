import { useState } from "react";
import type { FormEvent, ReactNode } from "react";
import {
  AlertTriangle,
  Braces,
  CheckCircle2,
  Clock,
  Loader2,
  PlayCircle,
  Search,
  Wrench,
} from "lucide-react";

import { postJson } from "./api";
import type {
  AgentRunRequest,
  AgentRunResponse,
  AgentToolCall,
} from "./types";

type AgentPlaygroundProps = {
  onInspectRun: (runId: string) => void;
};

export default function AgentPlayground({
  onInspectRun,
}: AgentPlaygroundProps) {
  const [query, setQuery] = useState(
    "Calculate (142080 - 120000) / 120000 * 100"
  );
  const [retrievalProvider, setRetrievalProvider] = useState("hybrid");
  const [topK, setTopK] = useState(3);
  const [rerank, setRerank] = useState(true);
  const [candidateMultiplier, setCandidateMultiplier] = useState(3);
  const [maxSteps, setMaxSteps] = useState(5);
  const [documentId, setDocumentId] = useState("");
  const [experimentId, setExperimentId] = useState("");

  const [result, setResult] = useState<AgentRunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  async function runAgent(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!query.trim()) {
      setError("Query cannot be empty.");
      return;
    }

    const payload: AgentRunRequest = {
      query: query.trim(),
      retrieval_provider: retrievalProvider,
      top_k: topK,
      rerank,
      candidate_multiplier: candidateMultiplier,
      max_steps: maxSteps,
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

      const data = await postJson<AgentRunResponse, AgentRunRequest>(
        "/agents/run",
        payload
      );

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run agent");
    } finally {
      setRunning(false);
    }
  }

  return (
    <section id="agent-playground" className="agent-playground">
      <div className="agent-playground-header">
        <div>
          <p className="eyebrow">Agent Playground</p>
          <h2>Run tool-calling agents</h2>
          <p>
            Test calculator, document-search, SQL-style, and mock API workflows
            through the agent runner, then inspect the full trace.
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

      <form className="agent-form" onSubmit={runAgent}>
        <label className="query-box">
          <span>Agent task</span>
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={3}
            placeholder="Ask the agent to calculate, search documents, or call a tool..."
          />
        </label>

        <div className="agent-example-row">
          <ExampleButton
            label="Calculator"
            onClick={() =>
              setQuery("Calculate (142080 - 120000) / 120000 * 100")
            }
          />
          <ExampleButton
            label="Order status"
            onClick={() => setQuery("Check order status for ORD-1001")}
          />
          <ExampleButton
            label="Customer plan"
            onClick={() => setQuery("What is the customer plan for CUST-1001?")}
          />
          <ExampleButton
            label="Document search"
            onClick={() =>
              setQuery(
                "What must customers provide when reporting a damaged product?"
              )
            }
          />
        </div>

        <div className="agent-controls">
          <ControlGroup label="Retriever">
            <select
              value={retrievalProvider}
              onChange={(event) => setRetrievalProvider(event.target.value)}
            >
              <option value="dense">Dense</option>
              <option value="bm25">BM25</option>
              <option value="hybrid">Hybrid</option>
            </select>
          </ControlGroup>

          <ControlGroup label="Top K">
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(event) => setTopK(safeNumber(event.target.value, 3))}
            />
          </ControlGroup>

          <ControlGroup label="Candidate Multiplier">
            <input
              type="number"
              min={1}
              max={10}
              value={candidateMultiplier}
              onChange={(event) =>
                setCandidateMultiplier(safeNumber(event.target.value, 3))
              }
            />
          </ControlGroup>

          <ControlGroup label="Max Steps">
            <input
              type="number"
              min={1}
              max={10}
              value={maxSteps}
              onChange={(event) => setMaxSteps(safeNumber(event.target.value, 5))}
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

          <button className="run-agent-button" type="submit" disabled={running}>
            {running ? (
              <>
                <Loader2 className="spin" size={18} />
                Running...
              </>
            ) : (
              <>
                <PlayCircle size={18} />
                Run Agent
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

      {result ? (
        <AgentResult result={result} onInspectRun={onInspectRun} />
      ) : null}
    </section>
  );
}

function AgentResult({
  result,
  onInspectRun,
}: {
  result: AgentRunResponse;
  onInspectRun: (runId: string) => void;
}) {
  return (
    <div className="agent-result">
      <div className="agent-result-top">
        <div>
          <p className="eyebrow">Agent Result</p>
          <h3>{result.run_id}</h3>
        </div>

        <div className="rag-badge-row">
          <span className={result.status === "completed" ? "badge success" : "badge danger"}>
            {result.status}
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

      <div className="agent-summary-grid">
        <MetricBox
          label="Latency"
          value={`${result.total_latency_ms} ms`}
          icon={<Clock size={17} />}
        />
        <MetricBox
          label="Tool Calls"
          value={formatNumber(result.tool_calls.length)}
          icon={<Wrench size={17} />}
        />
        <MetricBox
          label="Successful Calls"
          value={formatNumber(
            result.tool_calls.filter((toolCall) => toolCall.success).length
          )}
          icon={<CheckCircle2 size={17} />}
        />
      </div>

      <div className="agent-answer-card">
        <h4>Final Answer</h4>
        <p>{result.final_answer}</p>
      </div>

      <ResultSection
        title={`Tool Calls (${result.tool_calls.length})`}
        icon={<Wrench size={18} />}
      >
        {result.tool_calls.length === 0 ? (
          <p className="muted-text">No tool calls were recorded.</p>
        ) : (
          <div className="tool-call-list">
            {result.tool_calls.map((toolCall, index) => (
              <ToolCallCard
                key={`${toolCall.tool_name}-${index}`}
                toolCall={toolCall}
                index={index}
              />
            ))}
          </div>
        )}
      </ResultSection>

      <ResultSection title="Agent Metadata" icon={<Braces size={18} />}>
        <JsonBlock value={result.metadata} />
      </ResultSection>
    </div>
  );
}

function ToolCallCard({
  toolCall,
  index,
}: {
  toolCall: AgentToolCall;
  index: number;
}) {
  return (
    <article className="tool-call-card">
      <div className="tool-call-top">
        <div>
          <strong>
            #{index + 1} {toLabel(toolCall.tool_name)}
          </strong>
          <p>{toolCall.tool_name}</p>
        </div>

        <span className={toolCall.success ? "badge success" : "badge danger"}>
          {toolCall.success ? "success" : "failed"}
        </span>
      </div>

      {toolCall.error_message ? (
        <div className="rag-warning">
          <AlertTriangle size={17} />
          <span>{toolCall.error_message}</span>
        </div>
      ) : null}

      <div className="agent-json-grid">
        <JsonBlock title="Input" value={toolCall.input_data} />
        <JsonBlock title="Output" value={toolCall.output_data} />
      </div>
    </article>
  );
}

function ExampleButton({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <button className="example-button" type="button" onClick={onClick}>
      {label}
    </button>
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

function MetricBox({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: ReactNode;
}) {
  return (
    <div className="rag-metric-box">
      <span>
        {icon}
        {label}
      </span>
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

function JsonBlock({
  title,
  value,
}: {
  title?: string;
  value: Record<string, unknown>;
}) {
  return (
    <div className="json-block json-block-full">
      {title ? <p>{title}</p> : null}
      <pre>{JSON.stringify(value || {}, null, 2)}</pre>
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