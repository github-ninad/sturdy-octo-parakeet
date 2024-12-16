from phi.agent import Agent, AgentMemory
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.tools.pubmed import PubmedTools
from phi.tools.python import PythonTools
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.reranker.cohere import CohereReranker
from rich.pretty import pprint
from phi.memory.db.sqlite import SqliteMemoryDb
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
from phi.tools.calculator import Calculator

from config.settings import settings

# from agents import knowledge_base

#############################################
# MODEL SETUP
#############################################
manager_model = settings.AGENT_CONFIG.get('manager_model')
lead_model = settings.AGENT_CONFIG.get('lead_model')
worker_model = settings.AGENT_CONFIG.get('worker_model')

# Shared settings for memory
memory_settings = AgentMemory(
    db=SqliteMemoryDb(
        table_name="agent_memory",
        db_file="tmp/agent_memory.db",  # Memory database file
    ),
    create_user_memories=True,  # Enable personalized memory
    update_user_memories_after_run=True,  # Update after each run
    create_session_summary=True,  # Store session summaries
    update_session_summary_after_run=True,  # Update summaries after each run
)

# Shared settings for session storage
storage_settings = SqlAgentStorage(
    table_name="agent_sessions",
    db_file="tmp/agent_storage.db",  # Session storage database file
)

knowledge_base = None
#############################################
# KNOWLEDGE BASE SETUP (Uncomment and configure as needed)
#############################################
# knowledge_base = PDFKnowledgeBase(
#     path="myhealth-suraksha---pww.pdf",
#     vector_db=LanceDb(
#         table_name="insurance_policies",
#         uri=settings.LANCEDB_URI,
#         search_type=SearchType.hybrid,
#         reranker=CohereReranker(model="rerank-multilingual-v3.0"),
#         embedder=OpenAIEmbedder(model="text-embedding-3-small")
#     ),
# )
# knowledge_base.load()
#############################################
# RULES EXTRACTION & CODING TEAMS
#############################################
policy_extractor = Agent(
    name="PolicyExtractionAgent",
    role="Extracts structured rules from policy PDF",
    model=worker_model,
    reasoning=True,
    search_knowledge=True,
    tools=[PubmedTools()],
    show_tool_calls=True,
    instructions=[
        "Parse policy document for eligibility criteria, sub-limits, exclusions",
        "Create structured (JSON-like) rules",
        "Ensure data consistency and completeness"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

rule_coder = Agent(
    name="RuleCoderAgent",
    role="Convert extracted rules into executable logic",
    model=worker_model,
    reasoning=True,
    instructions=[
        "From structured rules, generate Python code functions",
        "Functions handle eligibility, sub-limits, co-payments, etc.",
        "Test code logic with sample scenarios"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

rules_team = Agent(
    team=[policy_extractor, rule_coder],
    model=lead_model,
    instructions=[
        "Combine extracted rules and coded logic into a ready-to-use module.",
        "Ensure no conflicts or missing rules."
    ],
    reasoning=True,
    respond_directly=True,
    markdown=True
)

#############################################
# MEDICAL TEAM
#############################################
medical_analyzer = Agent(
    name="Medical Protocol Analyzer",
    role="Analyze medical treatment protocols",
    model=worker_model,
    reasoning=True,
    tools=[PubmedTools()],
    show_tool_calls=True,
    instructions=[
        "Predict ICD-10 codes for given diagnoses",
        "Assess necessity of treatments and medications",
        "Search for evidences for such diagnosis and treatment",
        "Validate length of stay and unusual patterns"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

clinical_auditor = Agent(
    name="Clinical Documentation Specialist",
    role="Check documentation integrity",
    model=worker_model,
    tools=[PubmedTools(), DuckDuckGo()],
    show_tool_calls=True,
    reasoning=True,
    instructions=[
        "Review completeness of clinical documentation",
        "Validate diagnosis & procedure coding",
        "Identify any documentation gaps"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

medical_team = Agent(
    team=[medical_analyzer, clinical_auditor],
    model=lead_model,
    tools=[PubmedTools(), DuckDuckGo(), Newspaper4k()],
    show_tool_calls=True,
    instructions=[
        "Produce a comprehensive medical assessment report with ICD validation",
        "Ensure clarity and tabular summary if needed",
    ],
    reasoning=True,
    respond_directly=True,
    markdown=True
)

#############################################
# FRAUD & COMPLIANCE TEAM
#############################################
fraud_analyzer = Agent(
    name="Fraud Pattern Detector",
    role="Search and Identify Identify potential fraud patterns",
    model=worker_model,
    reasoning=True,
    tools=[DuckDuckGo(), Newspaper4k()],
    show_tool_calls=True,
    instructions=[
        "Search and Check for common fraud indicators for similar type of cases",
        "Analyze billing patterns",
        "Verify no duplicate or suspicious claims",
        "Search for Fraud incidences for something similar, and check if it's applicable here"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

compliance_checker = Agent(
    name="Compliance Validator",
    role="Check regulatory and policy compliance",
    model=worker_model,
    reasoning=True,
    instructions=[
        "Ensure no policy violations",
        "Check for proper authorizations",
        "Document any compliance issues"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

fraud_team = Agent(
    team=[fraud_analyzer, compliance_checker],
    model=lead_model,
    instructions=[
        "Produce a fraud & compliance analysis report",
        "Use tables for summarizing checks"
    ],
    reasoning=True,
    markdown=True,
    respond_directly=True,
    structured_outputs=True
)

#############################################
# FINANCIAL TEAM
#############################################
cost_analyzer = Agent(
    name="Cost Analysis Specialist",
    role="Compare claimed costs with standard rates",
    model=worker_model,
    tools=[DuckDuckGo(), Newspaper4k()],
    show_tool_calls=True,
    reasoning=True,
    instructions=[
        "Review procedure costs against standard or customary rates",
        "Check billing code accuracy",
        "Apply room rent caps if needed"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

benefit_calculator = Agent(
    name="Benefit Calculator",
    role="Calculate final payable amount",
    model=worker_model,
    reasoning=True,
    search_knowledge=True,
    tools=[
        Calculator(
            add=True,
            subtract=True,
            multiply=True,
            divide=True,
            exponentiate=True,
            factorial=True,
            is_prime=True,
            square_root=True,
        ), PythonTools()
    ],
    show_tool_calls=True,
    instructions=[
        "Use rules from RuleCoderAgent to apply sub-limits and co-payments",
        "Produce a stepwise calculation table",
        "Derive final eligible amount"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

financial_team = Agent(
    team=[cost_analyzer, benefit_calculator],
    model=lead_model,
    instructions=[
        "Generate a detailed financial analysis with stepwise calculations",
        "Include expense breakdown and final eligible amount"
    ],
    reasoning=True,
    markdown=True,
    respond_directly=True,
    structured_outputs=True
)

#############################################
# CALCULATION & VALIDATION TEAMS (TOT)
#############################################
calculation_agent = Agent(
    name="CalculationAgent",
    role="Tree-of-Thoughts based calculation strategy selection",
    model=manager_model,
    reasoning=True,
    tools=[
        Calculator(
            add=True,
            subtract=True,
            multiply=True,
            divide=True,
            exponentiate=True,
            factorial=True,
            is_prime=True,
            square_root=True,
        ), PythonTools()
    ],
    show_tool_calls=True,
    instructions=[
        "Propose multiple calculation strategies (Path A, Path B, etc.)",
        "Evaluate each for compliance, arithmetic correctness, and alignment with policy",
        "Select the best path with a clear rationale"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

calculation_validator = Agent(
    name="CalculationValidatorAgent",
    role="Cross-check chosen calculation path",
    model=worker_model,
    reasoning=True,
    tools=[
        Calculator(
            add=True,
            subtract=True,
            multiply=True,
            divide=True,
            exponentiate=True,
            factorial=True,
            is_prime=True,
            square_root=True,
        ), PythonTools()
    ],
    show_tool_calls=True,
    instructions=[
        "Verify arithmetic steps of the chosen calculation path",
        "Ensure consistency with policy rules",
        "Document validation in a table"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

#############################################
# DISCREPANCY ANALYZER
#############################################
discrepancy_analyzer = Agent(
    name="DiscrepancyAnalyzerAgent",
    role="Identify inter-team inconsistencies",
    model=worker_model,
    reasoning=True,
    instructions=[
        "Compare outputs from medical, fraud, compliance, financial, and review teams",
        "Flag contradictions or inconsistencies",
        "Suggest resolutions"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

#############################################
# EXPLAINABILITY & AUDIT TRAIL
#############################################
explainability_agent = Agent(
    name="ExplainabilityAgent",
    role="Produce a simple narrative explanation",
    model=worker_model,
    reasoning=True,
    instructions=[
        "Explain final decision in non-technical terms",
        "Highlight key policy clauses influencing the outcome"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

audit_trail_agent = Agent(
    name="AuditTrailAgent",
    role="Maintain chronological decision log",
    model=worker_model,
    reasoning=True,
    instructions=[
        "Record each step (timestamp, action, agent)",
        "Present audit log in a table"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

#############################################
# THINKER & GRADER AGENTS FOR TOT
#############################################
thinker_agent = Agent(
    name="ThinkerAgent",
    role="Generate multiple reasoning chains (Paths) for complex steps",
    model=manager_model,
    reasoning=True,
    instructions=[
        "When invoked, produce multiple possible reasoning paths to solve complex calculation scenario",
        "Label them Path A, Path B, etc.",
        "Each path should have a detailed step-by-step approach"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

grader_agent = Agent(
    name="GraderAgent",
    role="Evaluate multiple reasoning chains and pick the best",
    model=manager_model,
    reasoning=True,
    instructions=[
        "Compare reasoning paths from ThinkerAgent or CalculationAgent",
        "Select the best (most policy-compliant, accurate) path",
        "Document reasons for selection"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

#############################################
# FINAL REVIEW TEAM
#############################################
policy_reviewer = Agent(
    name="Policy Compliance Reviewer",
    role="Ensure final decision aligns with policy",
    model=worker_model,
    reasoning=True,
    search_knowledge=True,
    instructions=[
        "Check waiting periods, coverage terms",
        "Document references to policy sections"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

quality_auditor = Agent(
    name="Quality Assurance Specialist",
    role="Check overall quality and consistency",
    model=worker_model,
    reasoning=True,
    instructions=[
        "Validate final output formatting",
        "Check correctness of tables and data",
        "Ensure final markdown readability"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

review_team = Agent(
    team=[policy_reviewer, quality_auditor],
    model=lead_model,
    reasoning=True,
    markdown=True,
    structured_outputs=True,
    respond_directly=True,
    instructions=[
        "Produce final comprehensive review",
        "Summarize findings and ensure high-quality presentation"
    ]
)

#############################################
# MASTER TEAM
#############################################
master_team = Agent(
    team=[
        rules_team,
        medical_team,
        fraud_team,
        financial_team,
        calculation_agent,
        calculation_validator,
        discrepancy_analyzer,
        review_team,
        explainability_agent,
        audit_trail_agent,
        thinker_agent,
        grader_agent
    ],
    model=manager_model,
    instructions=[
        "Integrate all steps: policy extraction, medical check, fraud & compliance, financial TOT calculations, validation, discrepancy checks, final review.",
        "Use ThinkerAgent and CalculationAgent for generating multiple solution paths.",
        "Use GraderAgent to pick the best reasoning path.",
        "At the end, produce a final Markdown report with:",
        "- A 'Summary Report' with key claim details and results",
        "- A 'Medical Assessment' section with ICD codes, necessity validation, documentation completeness in a table",
        "- A 'Fraud & Compliance' section with a table of checks",
        "- A 'Financial Evaluation' with expense breakdown, sub-limits application table, final eligible amount",
        "- A 'Compliance Validation' section confirming arithmetic and policy alignment",
        "- A 'Claim Decision' stating final approved amount and referencing policy",
        "- A 'Decision Explanation' describing the chosen TOT path and reasoning steps",
        "- An 'Audit Log' table with timestamps and responsible agents",
        "- 'Recommendations' for improvements",
        "Ensure maximum transparency, detail, and readability."
    ],
    knowledge=knowledge_base,
    markdown=True,
    reasoning=True,
    structured_outputs=True
)


#############################################
# CLAIM PROCESSING FUNCTION
#############################################
def process_claim(claim_details):
    """
    Process the claim through the entire pipeline:
    1. Extract & code rules
    2. Medical assessment
    3. Fraud & compliance check
    4. Financial analysis (with multiple TOT strategies)
    5. Validate chosen calculation path
    6. Check for discrepancies across teams
    7. Final review and ensure policy compliance
    8. Explain the final decision and maintain an audit trail
    9. Present everything in a clear, markdown-structured report with tables and bullet points

    Additional Requirements:
    - Include TOT reasoning steps from CalculationAgent or ThinkerAgent
    - Show the final chosen reasoning path (selected by GraderAgent)
    - Highlight references to policy clauses and numeric details from the claim
    - Ensure a patient-friendly explanation
    """

    prompt = f"""
    Claim Details: {claim_details}

    Required Assessments and Output Format:

    ### Medical Assessment
    - Include ICD code prediction & validation for the given condition.
    - Confirm medical necessity for the procedure(s).
    - Check documentation completeness; present a table of required vs. received documents.

    ### Fraud & Compliance Check
    - List potential fraud checks in a table (e.g., duplicate claims, suspicious billing patterns).
    - Confirm compliance with policy terms and identify any violations.

    ### Financial Analysis (with TOT)
    - Present an expense breakdown in a table (item-wise costs).
    - Then Show a detailed step wise calculation trail with table for each section
        - Table of each billable item and its initial cost.
        - Apply each relevant policy rule step-by-step.
        - Show room rent capping, sub-limits, co-payments if applicable, and final eligibility per line item.
        - Reference the exact policy clause for each applied rule.
        - Conclude with the final eligible amount.
    - Step wise, Show multiple calculation strategies (if any) and indicate which one is chosen by the GraderAgent.
    - Finalize the eligible claim amount after applying all rules. All rules should be shown.

    ### Compliance Validation
    - Confirm arithmetic accuracy with rationale and that calculations align with extracted rules.
    - State that no discrepancies found (if accurate).

    ### Claim Decision
    - State the final approved amount.
    - Reference specific policy clauses justifying this amount.
    - Provide a short rationale.

    ### Decision Explanation
    - Describe the TOT reasoning: paths considered, why the chosen path was selected.
    - Explain in simple terms for the claimant.

    ### Audit Log
    - Provide a chronological table of each step (timestamp, action, agent).

    ### Recommendations
    - Suggest improvements for internal processes or clarity.

    Formatting:
    - Use Markdown headings (e.g., `## Summary Report`, `### Medical Assessment`).
    - Use tables where dense information is present.
    - Use bullet points for clarity.
    - Ensure all amounts and references are consistent.

    Now, integrate all agent outputs, finalize the reasoning, and produce the final Markdown report.

    Think Step wise and follow each an every step in detail as mentioned in the section.
    """
    return master_team.print_response(prompt, stream=True, show_full_reasoning=True)


def process_claim_detailed(claim_details):
    """
    Process the claim through the entire pipeline:
    1. Extract & code rules.
    2. Perform a detailed medical assessment with ICD validation, necessity checks, and documentation completeness.
    3. Conduct a fraud & compliance check with a table of potential fraud indicators and compliance checks.
    4. Execute a financial analysis with step-by-step calculations using multiple Tree-of-Thoughts (TOT) strategies.
    5. Validate the chosen calculation path for accuracy and adherence to policy rules.
    6. Check for discrepancies across all teams and resolve them.
    7. Perform a final review and ensure policy compliance.
    8. Generate a detailed explanation of the final decision using clear reasoning steps.
    9. Maintain an audit trail and produce actionable recommendations.

    Output Requirements:
    - Markdown-structured report.
    - Detailed, step-wise, and table-driven analysis of each section.
    - Include TOT reasoning steps, multiple strategies, and the final chosen path.
    - Reference policy clauses explicitly where rules are applied.
    - Ensure accuracy and patient-friendliness in explanations.
    """

    prompt = f"""
    Claim Details: {claim_details}

    Instruction: Do a detailed Adjudication based on below expectations of required assessment and format. Let's Think step by step

    Required Assessments and Output Format:

    ## Summary Report
    - Provide an overview of key claim details, including:
      - Patient name, case reference, claimed amount, and final approved amount.
      - High-level observations (e.g., emergency status, policy adherence, and major findings).

    ## Medical Assessment
    - ICD Code Prediction & Validation:
      - Predict ICD codes for the given condition.
      - Present a table of predicted ICD codes and their descriptions.
    - Treatment Necessity:
      - Verify medical necessity for the procedures and treatments.
      - Provide a very detailed rationale for necessity decisions.
    - Documentation Completeness:
      - Present a table listing all required documents (e.g., discharge summary, bills, receipts, authorization forms).
      - Include columns for whether each document was received, completeness status, and gaps identified (if any).

    ## Fraud & Compliance Check
    - Fraud Indicators:
      - List potential fraud checks applied in table format even if not flagged any (e.g., duplicate claims, suspicious patterns, invalid provider credentials).
      - Provide a table with each fraud indicator, evidence found, and comments.
    - Compliance:
      - Verify adherence to policy terms (e.g., sub-limits, exclusions, waiting periods).
      - Present a table summarizing each compliance check, its status (compliant or non-compliant), and detailed comments.

    ## Financial Analysis
    - Expense Breakdown:
      - Provide a table of all itemized expenses (e.g., room charges, procedure costs, investigations, medications).
      - Include columns for item name, claimed amount, applicable policy rules, and eligible amount.
    - Step-Wise Calculation Trail:
      - For each expense item, show how policy rules (e.g., room rent capping, sub-limits, co-payments) are applied.
      - Present each calculation step in a detailed table with references to the exact policy clauses.
    - Tree-of-Thoughts (TOT) Reasoning:
      - Generate multiple calculation strategies for determining the eligible amount.
      - Label each strategy (e.g., Path A, Path B) and provide step-by-step calculations for each in a table format.
      - Include a table summarizing the differences between strategies (e.g., total eligible amount, compliance level).
    - Final Eligible Amount:
      - Indicate the final chosen strategy and explain why it was selected (referencing the GraderAgent’s evaluation).
      - Present the final eligible amount in a table with a summary of the applied rules. This has to be extremely detailed.

    ## Compliance Validation
    - Arithmetic Verification:
      - Confirm that all calculations (e.g., deductions, co-payments, sub-limits) are arithmetically accurate.
      - Provide a table showing each calculation step, its inputs, and its verified outputs.
    - Policy Rule Alignment:
      - State whether all calculations align with the extracted policy rules.
      - Highlight any discrepancies (if found) and how they were resolved.

    ## Claim Decision
    - Final Approved Amount:
      - State the final approved amount in a clear and bold manner.
      - Reference specific policy clauses justifying the decision.
    - Provide a concise rationale summarizing why the claim was approved or rejected.

    ## Decision Explanation
    - Tree-of-Thoughts Reasoning:
      - Describe all strategies considered (e.g., Path A, Path B) and the rationale for selecting the chosen path.
      - Explain the reasoning in clear, simple terms for patient understanding.
    - Highlight key policy clauses and rules influencing the decision.

    ## Audit Log
    - Present a chronological table of all major actions taken during the claim processing, including:
      - Timestamp, action, agent responsible, and any additional comments.

    ## Recommendations
    - Suggest process improvements based on observations during the claim processing (e.g., automation of repetitive tasks, clearer policy documentation).
    - Provide actionable insights for reducing errors or inefficiencies.

    ## Formatting Requirements
    - Use Markdown headings for clear separation of sections.
    - Use tables for dense information (e.g., expenses, compliance checks, calculations).
    - Use bullet points for concise summaries and observations.
    - Ensure all references to policy rules and numeric details are consistent and accurate.

    Now think step wise, process the claim using all available agents, integrate outputs from each stage, and produce the final Markdown report.
    """
    return master_team.run(prompt, stream=True, stream_intermediate_steps=True)

def get_detailed_adjudication_prompt(claim_details):
    prompt = f"""
    Claim Details: {claim_details}

    Instruction: Do a detailed Adjudication based on below expectations of required assessment and format. Let's Think step by step

    Required Assessments and Output Format:

    ## Summary Report
    - Provide an overview of key claim details, including:
      - Patient name, case reference, claimed amount, and final approved amount.
      - High-level observations (e.g., emergency status, policy adherence, and major findings).

    ## Medical Assessment
    - ICD Code Prediction & Validation:
      - Predict ICD codes for the given condition.
      - Present a table of predicted ICD codes and their descriptions.
    - Treatment Necessity:
      - Verify medical necessity for the procedures and treatments.
      - Provide a very detailed rationale for necessity decisions.
    - Documentation Completeness:
      - Present a table listing all required documents (e.g., discharge summary, bills, receipts, authorization forms).
      - Include columns for whether each document was received, completeness status, and gaps identified (if any).

    ## Fraud & Compliance Check
    - Fraud Indicators:
      - List potential fraud checks applied in table format even if not flagged any (e.g., duplicate claims, suspicious patterns, invalid provider credentials).
      - Provide a table with each fraud indicator, evidence found, and comments.
    - Compliance:
      - Verify adherence to policy terms (e.g., sub-limits, exclusions, waiting periods).
      - Present a table summarizing each compliance check, its status (compliant or non-compliant), and detailed comments.

    ## Financial Analysis
    - Expense Breakdown:
      - Provide a table of all itemized expenses (e.g., room charges, procedure costs, investigations, medications).
      - Include columns for item name, claimed amount, applicable policy rules, and eligible amount.
    - Step-Wise Calculation Trail:
      - For each expense item, show how policy rules (e.g., room rent capping, sub-limits, co-payments) are applied.
      - Present each calculation step in a detailed table with references to the exact policy clauses.
    - Tree-of-Thoughts (TOT) Reasoning:
      - Generate multiple calculation strategies for determining the eligible amount.
      - Label each strategy (e.g., Path A, Path B) and provide step-by-step calculations for each in a table format.
      - Include a table summarizing the differences between strategies (e.g., total eligible amount, compliance level).
    - Final Eligible Amount:
      - Indicate the final chosen strategy and explain why it was selected (referencing the GraderAgent’s evaluation).
      - Present the final eligible amount in a table with a summary of the applied rules. This has to be extremely detailed.

    ## Compliance Validation
    - Arithmetic Verification:
      - Confirm that all calculations (e.g., deductions, co-payments, sub-limits) are arithmetically accurate.
      - Provide a table showing each calculation step, its inputs, and its verified outputs.
    - Policy Rule Alignment:
      - State whether all calculations align with the extracted policy rules.
      - Highlight any discrepancies (if found) and how they were resolved.

    ## Claim Decision
    - Final Approved Amount:
      - State the final approved amount in a clear and bold manner.
      - Reference specific policy clauses justifying the decision.
    - Provide a concise rationale summarizing why the claim was approved or rejected.

    ## Decision Explanation
    - Tree-of-Thoughts Reasoning:
      - Describe all strategies considered (e.g., Path A, Path B) and the rationale for selecting the chosen path.
      - Explain the reasoning in clear, simple terms for patient understanding.
    - Highlight key policy clauses and rules influencing the decision.

    ## Audit Log
    - Present a chronological table of all major actions taken during the claim processing, including:
      - Timestamp, action, agent responsible, and any additional comments.

    ## Recommendations
    - Suggest process improvements based on observations during the claim processing (e.g., automation of repetitive tasks, clearer policy documentation).
    - Provide actionable insights for reducing errors or inefficiencies.

    ## Formatting Requirements
    - Use Markdown headings for clear separation of sections.
    - Use tables for dense information (e.g., expenses, compliance checks, calculations).
    - Use bullet points for concise summaries and observations.
    - Ensure all references to policy rules and numeric details are consistent and accurate.

    Now, process the claim using all available agents, integrate outputs from each stage, and produce the final Markdown report.
    """
    return prompt


def generate_calculation_breakdown(claim_amount):
    """
    Provide a highly detailed calculation breakdown:
    - Extract relevant policy rules (e.g., room rent limit, surgical package rates).
    - Show step-by-step calculation in a table, referencing policy sections.
    - Include sub-limits, co-pays, package rates, and final eligible amount.
    - Make the output in Markdown, clearly indicating each step and how the final number is derived.
    """
    prompt = f"""
    Generate a detailed calculation breakdown for:
    Claim Amount: {claim_amount}

    Required:
    - Table of each billable item and its initial cost.
    - Apply each relevant policy rule step-by-step.
    - Show room rent capping, sub-limits, co-payments if applicable, and final eligibility per line item.
    - Reference the exact policy clause for each applied rule.
    - Conclude with the final eligible amount.
    """
    return benefit_calculator.print_response(prompt, stream=True, show_full_reasoning=True)
