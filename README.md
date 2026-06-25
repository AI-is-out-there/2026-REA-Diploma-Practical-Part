# Main idea
The code implements a telemedicine triage agent that uses the Mistral LLM to analyze patient complaints by retrieving structured clinical guidelines and traversing a medical knowledge graph to provide evidence-based summaries and operator tips. The code utilizes information from the Russian Ministry of Health's clinical recommendations portal (`https://cr.minzdrav.gov.ru`) through a structured, hardcoded database rather than live web scraping.

# 1. Code review 

a.  **Large Language Model (LLM) Integration**: Uses `ChatMistralAI` via LangChain to process natural language queries from doctors.
b.  **Clinical Requirements Database**: A structured dictionary containing evidence-based clinical guidelines for chronic conditions (e.g., Hypertension, Type 2 Diabetes, Stroke, Heart Failure, Asthma), including diagnostic criteria, treatment protocols, and target metrics.
c.  **Medical Knowledge Graph**: Built using `NetworkX`, this graph maps complex relationships between medical entities (diseases, treatments, diets), clinical guidelines, and specific medical claims. Crucially, it assigns "verdicts" (supported/refuted) to common medical myths or claims based on established guidelines.
d.  **Agentic Workflow**: Utilizes LangChain's `AgentExecutor` and tools, allowing the AI to autonomously query the databases and knowledge graph to retrieve accurate, cited medical information.
e.  **Graph Visualization**: Includes code to visualize the knowledge graph using `Matplotlib`, mapping nodes (claims, guidelines, entities) and edges (relationships like "contradicts", "supports", "requires") to help visualize medical logic.


# 2. Case Study: AI-Assisted Telemedicine Consultation 

## Context and Scenario
**Setting:** A remote telemedicine video consultation between a primary care physician (Dr. Ivanov) and a 58-year-old patient, Ivan, who has recently been diagnosed with both **Hypertension** and **Type 2 Diabetes**. 
**The Challenge:** During the 15-minute consultation, Ivan is anxious and shares several conflicting health tips he read on social media. Dr. Ivanov needs to quickly establish a treatment plan, set target health metrics, and correct dangerous medical misinformation without spending excessive time searching through external medical literature.

## 2. Application of the AI Assistant
Dr. Ivanov has the **Telemed-conf-AI-assistant** running in the background, integrated into the telemedicine platform's dashboard. 

### Step 1: Real-Time Guideline Retrieval
Dr. Ivanov types a quick prompt into the assistant: *"What are the target blood pressure and HbA1c goals for a patient with both hypertension and diabetes?"*
*   **Under the Hood:** The LangChain agent queries the `CLINICAL_REQUIREMENTS_DB`.
*   **AI Response:** The assistant instantly retrieves the exact metrics: Target BP <130/80 mmHg (due to cardiovascular risk) and HbA1c <7%. It also suggests first-line pharmacotherapies (e.g., ACE inhibitors for BP, Metformin for diabetes).

### Step 2: Myth-Busting via the Knowledge Graph
Ivan interrupts, saying: *"My neighbor told me that since I have diabetes, I can completely cure it just by following a strict keto diet and stop taking Metformin."*
Dr. Ivanov asks the AI: *"Verify claim: Type 2 diabetes can be completely cured by diet alone."*
*   **Under the Hood:** The AI searches the `MEDICAL_KNOWLEDGE_GRAPH`. It locates `claim_2` ("Diabetes type 2 can be completely cured by diet"), checks its connections, and sees it has a verdict of **"refuted"** with high confidence (0.9), contradicting `guideline_2`.
*   **AI Response:** The assistant alerts Dr. Ivanov: *"This claim is **refuted** by current clinical guidelines. While diet is crucial for management, Type 2 Diabetes requires ongoing medical therapy (like Metformin) and monitoring. Diet alone is not a complete cure."* Dr. Ivanov uses this evidence to gently correct John's misconception.

### Step 3: Visualizing the Care Plan
To help Ivan understand why he needs multiple medications and lifestyle changes, Dr. Ivanov clicks the **"Show Care Graph"** button.
*   **Under the Hood:** The system triggers the `NetworkX` and `Matplotlib` visualization code.
*   **AI Response:** A visual graph appears on John's screen. It shows nodes for "Hypertension" and "Diabetes" connected to their respective required treatments ("Medications", "Dietary restrictions") and monitoring requirements. The visual aid helps the patient understand the interconnectedness of his conditions, increasing his adherence to the treatment plan.

# 2. Code Architecture 
### 1. Static Data Repository (`CLINICAL_REQUIREMENTS_DB`)
The code contains a Python dictionary named `CLINICAL_REQUIREMENTS_DB` that acts as a local cache of specific clinical guidelines. Each entry in this database corresponds to a major medical condition (e.g., hypertension, diabetes, stroke) and includes metadata directly sourced or summarized from the Ministry's portal:

*   **Source URL:** Each entry contains a specific `url` field pointing to the official page on `cr.minzdrav.gov.ru` (e.g., `"url": "https://cr.minzdrav.gov.ru/clin-rec/hypertension"`). This ensures that the operator can verify the source of the advice.
*   **Clinical Guidelines:** The `guidelines` list within each entry contains key diagnostic and treatment protocols extracted from the official documents. For example, for hypertension, it includes the definition of arterial hypertension (≥140/90 mm Hg) and target blood pressure values (<130/80 mm Hg).
*   **Metadata:** It tracks the `last_updated` date and `evidence_level` (e.g., "A"), which are critical for assessing the currency and reliability of the medical advice.

### 2. Retrieval Tool (`search_clinical_guidelines_ru`)
The code implements a LangChain tool named `search_clinical_guidelines_ru`. When the LLM agent identifies a medical condition in the patient's complaint, it invokes this tool. The tool performs the following steps:
*   **Keyword Matching:** It searches the keys and content of `CLINICAL_REQUIREMENTS_DB` for matches with the patient's condition (e.g., matching "high blood pressure" to the "hypertension" entry).
*   **Structured Output:** If a match is found, it returns a JSON object containing the relevant guidelines, the official URL from the Ministry's site, and the evidence level. This provides the LLM with authoritative, non-hallucinated facts to include in its response.

### 3. Integration with the Knowledge Graph
While the `CLINICAL_REQUIREMENTS_DB` holds the detailed text of the guidelines, the `MEDICAL_KNOWLEDGE_GRAPH` contains nodes representing these guidelines (e.g., `guideline_1`). 
*   **Cross-Referencing:** The graph links clinical entities (like "Hypertension") to their corresponding guideline nodes via `follows` relationships. 
*   **Fact-Checking:** Claims made by patients or found in unstructured data are linked to these guideline nodes via `supports` or `contradicts` edges. This allows the system to automatically flag misinformation by checking if a claim contradicts the official Ministry of Health recommendations stored in the graph.


# 3. System Architecture and Operational Workflow

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
