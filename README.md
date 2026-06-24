
# 1. System Architecture and Operational Workflow

The code implements a **Retrieval-Augmented Generation (RAG)** framework specialized for telemedicine triage and operator support. Unlike generic RAG systems that rely solely on vector similarity search over unstructured text, this system employs a **hybrid retrieval strategy** combining structured database lookups with symbolic knowledge graph traversal.

The operational workflow proceeds as follows:

1.  **Agent Orchestration:** The system utilizes the LangChain `AgentExecutor` pattern with a tool-calling agent. The LLM acts as a semantic router and synthesizer rather than a direct knowledge base. It receives a natural language patient complaint (e.g., "high blood pressure and headaches") and autonomously decides which tools to invoke.
2.  **Structured Retrieval Tools:**
    *   `search_clinical_guidelines_ru`: Performs exact or fuzzy string matching against a hardcoded dictionary (`CLINICAL_REQUIREMENTS_DB`) containing Russian Ministry of Health clinical guidelines. This ensures regulatory compliance and provides authoritative references.
    *   `query_clinical_knowledge_graph_ru`: Traverses the `MEDICAL_KNOWLEDGE_GRAPH` dictionary. It identifies relevant nodes (claims, entities, guidelines) via substring matching and retrieves associated edges (relationships). This tool returns structured JSON representing the local subgraph relevant to the query.
    *   `generate_operator_tips_from_graph`: A deterministic post-processing tool. It parses the JSON output from the graph query and applies rule-based logic to generate actionable advice for the human operator (e.g., flagging refuted claims or highlighting required treatments).
3.  **Synthesis:** The agent aggregates the outputs from these tools and generates a final response comprising a problem summary, evidence base, and operator tips.

### 2. Characteristics of the Mistral Chat Response

The system uses `mistral-small-latest` via the `ChatMistralAI` interface. In this architecture, the LLM's response generation is characterized by three distinct functional roles:

*   **Semantic Parsing and Tool Selection:** The model must accurately map vague patient descriptions ("suspects hypertension") to specific tool arguments (`condition="hypertension"`). The low temperature setting (`0.1`) is critical here to minimize hallucination during tool invocation and ensure deterministic parameter extraction.
*   **Contextual Integration:** The LLM does not generate medical advice from its parametric memory. Instead, it synthesizes the *retrieved* structured data. The prompt explicitly constrains the model to use the provided evidence base, reducing the risk of generating plausible but clinically incorrect information.
*   **Structured Output Formatting:** The model is instructed to adhere to a specific schema (Summary, Evidence Base, Operator Tips). This transforms the raw JSON outputs from the tools into a coherent, human-readable narrative suitable for a telemedicine dashboard.

### 3. Medical Knowledge Graph (MKG) Structure

The `MEDICAL_KNOWLEDGE_GRAPH` is implemented as a directed property graph tailored for clinical fact-checking and relationship mapping. Its ontology consists of three primary node types:

*   **Entities:** Clinical concepts categorized as `disease`, `treatment`, `medication`, or `dietary` (e.g., "Гипертония", "Соль"). These serve as the semantic anchors of the graph.
*   **Claims:** Propositional statements regarding medical facts (e.g., "Hypertension is treated only with medication"). Crucially, these nodes carry metadata attributes `verdict` (supported/refuted) and `confidence`, enabling automated misinformation detection.
*   **Guidelines:** References to authoritative clinical protocols with temporal metadata (`last_updated`), ensuring the system respects the currency of medical evidence.

The edge taxonomy defines semantic relationships such as `contradicts`, `supports`, `requires`, `treated_by`, and `mentions`. This explicit relational structure allows the system to perform multi-hop reasoning (e.g., identifying that a claim contradicts a guideline which in turn supports a specific entity).

### 4. Theoretical Justification for Graph-Based Representation

The integration of a Knowledge Graph (KG) addresses fundamental limitations of pure LLM approaches in high-stakes medical domains:

*   **Explainability and Provenance:** LLMs are opaque; their outputs cannot be easily traced to specific sources. The KG provides explicit, auditable paths between a patient's symptom and a clinical recommendation. When the system flags a claim as "refuted," it can point to the specific `contradicts` edge linking to a guideline, satisfying regulatory requirements for explainable AI in healthcare.
*   **Mitigation of Hallucination:** By grounding responses in a curated set of nodes and edges, the system constrains the LLM's generation space. The graph acts as a symbolic "guardrail," preventing the model from inventing relationships that do not exist in the validated knowledge base.
*   **Handling Contradictory Information:** Medical knowledge evolves, and outdated information persists in training data. Vector stores struggle to distinguish between current and deprecated facts. The KG explicitly models temporal validity and verdicts, allowing the system to programmatically suppress refuted claims regardless of their semantic similarity to the query.
*   **Relational Reasoning over Semantic Similarity:** Patient problems often require understanding causal or therapeutic links, not just lexical similarity. A vector search might retrieve documents about "headaches" when queried for "hypertension," but the KG explicitly encodes the `requires` or `treated_by` relationships, enabling precise retrieval of clinically relevant context that purely statistical models may miss.

In summary, the code demonstrates a neuro-symbolic approach where the LLM provides linguistic flexibility and interface capabilities, while the Knowledge Graph ensures clinical accuracy, traceability, and logical consistency.
