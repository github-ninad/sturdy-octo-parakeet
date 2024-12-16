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
knowledge_base = PDFKnowledgeBase(
    path="myhealth-suraksha---pww.pdf",
    vector_db=LanceDb(
        table_name="insurance_policies",
        uri=settings.LANCEDB_URI,
        search_type=SearchType.hybrid,
        reranker=CohereReranker(model="rerank-multilingual-v3.0"),
        embedder=OpenAIEmbedder(model="text-embedding-3-small")
    ),
)
knowledge_base.load()
#############################################
# RULES EXTRACTION & CODING TEAMS
#############################################
policy_extractor = Agent(
    name="PolicyExtractionAgent",
    role="Extract and structure insurance policy rules",
    model=worker_model,
    reasoning=True,
    search_knowledge=True,
    tools=[PubmedTools()],
    show_tool_calls=True,
    instructions=[
        "Extract all policy clauses related to claim eligibility",
        "Extract all sub-limits and their conditions",
        "Extract all exclusions and waiting periods",
        "Create a JSON structure with hierarchical rule organization",
        "Link each rule to its source policy section",
        "Validate each extracted rule for completeness",
    ],
    knowledge=knowledge_base,
    markdown=True,
)

rule_coder = Agent(
    name="RuleCoderAgent",
    role="Transform policy rules into executable code",
    model=worker_model,
    reasoning=True,
    instructions=[
        "Convert each JSON rule into a Python function",
        "Add input validation for each function",
        "Include error handling in functions",
        "Add docstrings with policy references",
        "Create test cases for each function",
    ],
    knowledge=knowledge_base,
    markdown=True,
)

rules_team = Agent(
    team=[policy_extractor, rule_coder],
    model=lead_model,
    instructions=[
        "Step 1: Direct PolicyExtractionAgent to extract rules section by section",
        "Step 2: Validate extracted rules for completeness",
        "Step 3: Send validated rules to RuleCoderAgent",
        "Step 4: Review coded functions for accuracy",
        "Step 5: Test complete ruleset with sample scenarios",
        "If any step fails, repeat the step with specific focus on failed components",
    ],
    reasoning=True,
    read_chat_history=True,
    respond_directly=True,
    markdown=True
)

#############################################
# MEDICAL TEAM
#############################################
medical_analyzer = Agent(
    name="Medical Protocol Analyzer",
    role="Validate medical necessity and coding",
    model=worker_model,
    reasoning=True,
    tools=[PubmedTools()],
    show_tool_calls=True,
    instructions=[
        "Search PubMed for standard treatment protocols for the diagnosis",
        "Match treatment provided against standard protocols",
        "Verify ICD-10 coding accuracy for primary diagnosis",
        "Verify ICD-10 coding for comorbidities",
        "Check if any treatments fall under experimental category",
        "Document all evidence sources from PubMed",
    ],
    knowledge=knowledge_base,
    markdown=True,
)

clinical_auditor = Agent(
    name="Clinical Documentation Specialist",
    role="Verify medical documentation completeness",
    model=worker_model,
    tools=[PubmedTools(), DuckDuckGo()],
    show_tool_calls=True,
    reasoning=True,
    instructions=[
        "Check for presence of admission notes",
        "Verify daily progress notes completeness",
        "Check for presence of investigation reports",
        "Verify operation notes if applicable",
        "Check discharge summary completeness",
        "List all missing documents",
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
        "Step 1: Direct MedicalAnalyzer to validate diagnosis and treatment",
        "Step 2: Review evidence found by MedicalAnalyzer",
        "Step 3: Direct ClinicalAuditor to check documentation",
        "Step 4: Review documentation gaps",
        "Step 5: Compile medical necessity confirmation",
        "Step 6: Generate final medical assessment report",
        "If medical necessity is unclear, repeat Step 1 with focus on specific treatments",
    ],
    reasoning=True,
    respond_directly=True,
    read_chat_history=True,
    markdown=True
)

#############################################
# FRAUD & COMPLIANCE TEAM
#############################################
fraud_analyzer = Agent(
    name="Fraud Pattern Detector",
    role="Investigate potential fraud patterns using data-driven analysis",
    model=worker_model,
    reasoning=True,
    tools=[DuckDuckGo(), Newspaper4k()],
    show_tool_calls=True,
    instructions=[
        "Step 1: Search for similar fraud cases in healthcare industry",
        "Step 2: Cross-reference claim patterns with known fraud indicators",
        "Step 3: Check provider history and credentials",
        "Step 4: Analyze billing codes for upcoding patterns",
        "Step 5: Check for duplicate submissions across claims",
        "Step 6: Verify service dates logic and sequence",
        "Step 7: Document all findings with evidence sources",
        "Create a fraud risk score based on findings"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

compliance_checker = Agent(
    name="Compliance Validator",
    role="Validate regulatory and policy compliance with evidence",
    model=worker_model,
    reasoning=True,
    instructions=[
        "Check pre-authorization requirements and documentation",
        "Verify waiting period compliance with dates",
        "Validate policy coverage status during service dates",
        "Check for excluded conditions or procedures",
        "Verify network provider compliance",
        "Document all non-compliant items with policy references",
        "Create compliance checklist with pass/fail status"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

fraud_team = Agent(
    team=[fraud_analyzer, compliance_checker],
    model=lead_model,
    instructions=[
        "Step 1: Direct FraudAnalyzer to conduct initial screening",
        "Step 2: Review fraud analysis findings",
        "Step 3: Direct ComplianceChecker for policy validation",
        "Step 4: Review compliance findings",
        "Step 5: Cross-validate fraud and compliance results",
        "Step 6: Generate comprehensive risk assessment",
        "If high risk identified, initiate detailed investigation",
        "Produce final report with evidence and recommendations"
    ],
    reasoning=True,
    markdown=True,
    respond_directly=True,
    read_chat_history=True,
    structured_outputs=True
)

#############################################
# FINANCIAL TEAM
#############################################
cost_analyzer = Agent(
    name="Cost Analysis Specialist",
    role="Analyze and validate claim costs against standards",
    model=worker_model,
    tools=[DuckDuckGo(), Newspaper4k()],
    show_tool_calls=True,
    reasoning=True,
    instructions=[
        "Search for standard treatment costs in the region",
        "Compare each line item against standard rates",
        "Check for package rate applicability",
        "Validate room rent against policy limits",
        "Analyze pharmacy charges against standard prices",
        "Flag items exceeding normal range with evidence",
        "Create cost variance report with justifications"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

benefit_calculator = Agent(
    name="Benefit Calculator",
    role="Execute precise benefit calculations with policy rules",
    model=worker_model,
    reasoning=True,
    search_knowledge=True,
    tools=[Calculator(), PythonTools()],
    show_tool_calls=True,
    instructions=[
        "Extract all applicable policy limits and sub-limits",
        "Calculate room rent capping adjustments",
        "Apply co-payment requirements per service",
        "Process deductibles and their applications",
        "Calculate proportionate deductions if any",
        "Apply package rate benefits if applicable",
        "Show each calculation step with formula used",
        "Document all policy clauses used in calculations"
    ],
    knowledge=knowledge_base,
    markdown=True,
)

financial_team = Agent(
    team=[cost_analyzer, benefit_calculator],
    model=lead_model,
    instructions=[
        "Step 1: Direct CostAnalyzer to validate expenses",
        "Step 2: Review cost analysis findings",
        "Step 3: Send validated costs to BenefitCalculator",
        "Step 4: Review calculation methodology",
        "Step 5: Validate final benefit amount",
        "If any anomalies found, repeat relevant step",
        "Generate detailed financial report with all calculations",
        "Include evidence sources and policy references"
    ],
    reasoning=True,
    markdown=True,
    respond_directly=True,
    read_chat_history=True,
    structured_outputs=True
)

#############################################
# CALCULATION & VALIDATION TEAMS (TOT)
#############################################
calculation_agent = Agent(
    name="CalculationAgent",
    role="Design and evaluate multiple calculation strategies",
    model=manager_model,
    reasoning=True,
    tools=[Calculator(), PythonTools()],
    show_tool_calls=True,
    instructions=[
        "Generate Path A: Standard benefit calculation approach",
        "Generate Path B: Package rate based approach",
        "Generate Path C: Proportionate calculation approach",
        "Document assumptions for each path",
        "List pros and cons of each approach",
        "Generate python code for calculation rules and execute the calculations",
        "Calculate outcomes for each path",
        "Compare results against policy terms",
        "Recommend best path with justification"
    ],
    knowledge=knowledge_base,
    search_knowledge=True,
    markdown=True,
)

calculation_validator = Agent(
    name="CalculationValidatorAgent",
    role="Validate calculations against policy terms",
    model=worker_model,
    reasoning=True,
    tools=[Calculator(), PythonTools()],
    show_tool_calls=True,
    instructions=[
        "Cross-check each arithmetic step",
        "Verify policy rule application sequence",
        "Validate sub-limit applications",
        "Check co-payment calculations",
        "Verify deductible applications",
        "Create validation checklist",
        "Document any discrepancies found",
        "Provide correction recommendations if needed"
    ],
    knowledge=knowledge_base,
    search_knowledge=True,
    read_chat_history=True,
    markdown=True,
)

#############################################
# DISCREPANCY ANALYZER
#############################################
discrepancy_analyzer = Agent(
    name="DiscrepancyAnalyzerAgent",
    role="Systematic analysis of inter-team findings consistency",
    model=worker_model,
    reasoning=True,
    tools=[PythonTools()],  # Added for structured comparison
    show_tool_calls=True,
    instructions=[
        "Create comparison matrix of all team findings",
        "Check numerical consistency across financial calculations",
        "Verify policy interpretation consistency between teams",
        "Compare medical necessity findings with policy terms",
        "Cross-validate fraud indicators with medical findings",
        "Document each discrepancy with source references",
        "Rate severity of each discrepancy (High/Medium/Low)",
        "Propose specific resolution steps for each issue"
    ],
    knowledge=knowledge_base,
    read_chat_history=True,
    markdown=True,
)

#############################################
# EXPLAINABILITY & AUDIT TRAIL
#############################################
explainability_agent = Agent(
    name="ExplainabilityAgent",
    role="Create clear, evidence-based explanations for decisions",
    model=worker_model,
    reasoning=True,
    tools=[DuckDuckGo()],  # Added for relevant examples
    show_tool_calls=True,
    instructions=[
        "Summarize key decision points in simple language",
        "Explain each policy rule application with examples",
        "Describe medical necessity decisions with context",
        "Clarify all calculations with simple breakdowns",
        "Provide rationale for any claim adjustments",
        "Link each decision to specific policy sections",
        "Create FAQ section for common questions",
        "Include glossary of technical terms used"
    ],
    knowledge=knowledge_base,
    read_chat_history=True,
    markdown=True,
)

audit_trail_agent = Agent(
    name="AuditTrailAgent",
    role="Maintain detailed chronological process documentation",
    model=worker_model,
    reasoning=True,
    tools=[PythonTools()],  # Added for timestamp handling
    show_tool_calls=True,
    instructions=[
        "Log each agent activation with timestamp",
        "Record all tool usage with outcomes",
        "Document all policy references accessed",
        "Track all calculation versions and changes",
        "Log all inter-agent communications",
        "Record all validation checks performed",
        "Document all discrepancy resolutions",
        "Create timeline of key decision points"
    ],
    knowledge=knowledge_base,
    read_chat_history=True,
    markdown=True,
)

#############################################
# THINKER & GRADER AGENTS FOR TOT
#############################################
thinker_agent = Agent(
    name="ThinkerAgent",
    role="Generate structured reasoning paths for complex decisions",
    model=manager_model,
    reasoning=True,
    tools=[PythonTools()],  # Added for logic structuring
    show_tool_calls=True,
    instructions=[
        "Break down complex decision into sub-components",
        "Generate three distinct reasoning approaches",
        "Document assumptions for each approach",
        "List pros and cons of each path",
        "Consider edge cases for each path",
        "Evaluate policy compliance of each path",
        "Document resource requirements per path",
        "Assess risks associated with each path"
    ],
    knowledge=knowledge_base,
    read_chat_history=True,
    markdown=True,
)

grader_agent = Agent(
    name="GraderAgent",
    role="Evaluate and select optimal reasoning paths",
    model=manager_model,
    reasoning=True,
    tools=[PythonTools()],  # Added for scoring system
    show_tool_calls=True,
    instructions=[
        "Create evaluation matrix for all paths",
        "Score each path on policy compliance",
        "Evaluate accuracy of calculations",
        "Check completeness of documentation",
        "Assess practicality of implementation",
        "Compare risk levels between paths",
        "Justify final path selection",
        "Document rejection reasons for other paths"
    ],
    knowledge=knowledge_base,
    read_chat_history=True,
    markdown=True,
)

#############################################
# FINAL REVIEW TEAM
#############################################
policy_reviewer = Agent(
    name="Policy Compliance Reviewer",
    role="Comprehensive policy alignment verification",
    model=worker_model,
    reasoning=True,
    search_knowledge=True,
    tools=[PythonTools()],  # Added for structured verification
    show_tool_calls=True,
    instructions=[
        "Verify all applicable policy sections checked",
        "Confirm waiting period calculations",
        "Validate coverage term applications",
        "Check all exclusion considerations",
        "Verify sub-limit applications",
        "Confirm documentation requirements met",
        "Validate authorization compliances",
        "Create policy reference matrix"
    ],
    knowledge=knowledge_base,
    read_chat_history=True,
    markdown=True,
)

quality_auditor = Agent(
    name="Quality Assurance Specialist",
    role="Ensure comprehensive quality standards",
    model=worker_model,
    reasoning=True,
    tools=[PythonTools()],  # Added for quality checks
    show_tool_calls=True,
    instructions=[
        "Check numerical accuracy in all calculations",
        "Verify formatting consistency across reports",
        "Validate all table structures and data",
        "Check completeness of all sections",
        "Verify citation accuracy and format",
        "Check language clarity and consistency",
        "Validate all cross-references",
        "Ensure proper markdown formatting"
    ],
    knowledge=knowledge_base,
    read_chat_history=True,
    markdown=True,
)

review_team = Agent(
    team=[policy_reviewer, quality_auditor],
    model=lead_model,
    reasoning=True,
    markdown=True,
    structured_outputs=True,
    read_chat_history=True,
    respond_directly=True,
    instructions=[
        "Step 1: Direct PolicyReviewer for initial compliance check",
        "Step 2: Review compliance findings",
        "Step 3: If compliance issues found, return to relevant team",
        "Step 4: Direct QualityAuditor for final review",
        "Step 5: Review quality findings",
        "Step 6: If quality issues found, coordinate fixes",
        "Step 7: Compile final comprehensive review",
        "Step 8: Generate executive summary with key findings",
        "Ensure all steps are documented with evidence"
    ]
)

#############################################
# MASTER TEAM
#############################################

master_team_v2 = Agent(
    team=[
        rules_team, medical_team, fraud_team, financial_team,
        calculation_agent, calculation_validator, discrepancy_analyzer,
        review_team, explainability_agent, audit_trail_agent,
        thinker_agent, grader_agent
    ],
    model=manager_model,
    instructions=[
        # Phase 1: Initial Setup and Rule Extraction
        "Direct rules_team to extract and structure policy rules",
        "Validate extracted rules completeness and accuracy",
        "If rule extraction incomplete, return to rules_team with specific gaps",

        # Phase 2: Parallel Analysis
        "Initiate parallel processing:",
        "- Direct medical_team for protocol analysis and documentation review",
        "- Direct fraud_team for pattern analysis and compliance check",
        "- Direct financial_team for initial cost analysis",
        "Wait for all parallel processes to complete",

        # Phase 3: Calculation Strategy
        "Engage thinker_agent to generate multiple calculation paths",
        "Direct grader_agent to evaluate and select optimal path",
        "Use calculation_agent to execute chosen path",
        "Have calculation_validator verify all computations",

        # Phase 4: Consistency Check
        "Direct discrepancy_analyzer to review all findings",
        "If discrepancies found:",
        "- Return to relevant team for resolution",
        "- Revalidate after corrections",

        # Phase 5: Final Processing
        "Direct review_team for final assessment",
        "Have explainability_agent create clear narrative",
        "Ensure audit_trail_agent has logged all steps",

        # Phase 6: Report Generation
        "Generate 'Agentic Adjudication Scrutiny Report' with following structure:",

        "Step 6.1: Executive Section",
        "- Compile executive summary with key decision points",
        "- Format policyholder and claim context",
        "- Present critical findings in clear format",

        "Step 6.2: Medical Analysis",
        "- Present diagnosis and treatment analysis narratively",
        "- Include ICD coding validation table",
        "- Document medical necessity evaluation",
        "- Show documentation assessment matrix",

        "Step 6.3: Risk Assessment",
        "- Detail fraud risk analysis narrative",
        "- Present critical risk indicators table",
        "- Document policy compliance validation",

        "Step 6.4: Financial Review",
        "- Rules and Calculation creation and execution",
        "- Include detailed calculations in appendix",
        "- Show expense analysis with benchmarks",
        "- Present calculation methodology paths",
        "- Explain selected approach rationale",

        "Step 6.5: Decision Documentation",
        "- Present amount summary with policy references",
        "- Provide clear decision rationale",
        "- Include patient-friendly explanation",

        "Step 6.6: Process Documentation",
        "- Include complete audit trail",
        "- Present recommendations matrix",
        "- Document special notes and insights",

        "Report Quality Requirements:",
        "- Ensure all assertions are evidence-backed",
        "- Verify all policy references",
        "- Validate calculation trails",
        "- Check narrative clarity",
        "- Confirm cross-reference accuracy",

        "Final Checks:",
        "- Review formatting consistency",
        "- Validate all data points",
        "- Ensure logical flow between sections",
        "- Verify completeness of all sections"
    ],
    knowledge=knowledge_base,
    read_chat_history=True,
    search_knowledge=True,
    markdown=True,
    reasoning=True
)

master_team = Agent(
    team=[
        rules_team,
        medical_team,
        fraud_team,
        financial_team,
        calculation_agent,
        calculation_validator,
        # discrepancy_analyzer,
        # review_team,
        # explainability_agent,
        # audit_trail_agent,
        thinker_agent,
        grader_agent
    ],
    model=manager_model,
    instructions=[
        # Phase 1: Policy Analysis & Rule Building
        "Extract all policy clauses and conditions using rules_team",
        "Convert extracted policy rules into structured, executable format",
        "Create hierarchical rule structure for systematic calculation processing",
        "Validate extracted rules against source policy document",
        "Document all rule dependencies and calculation parameters",
        "Map exclusions and special conditions to rule framework",

        # Phase 2: Data Collection & Validation
        "Direct medical_team to validate ICD codes using medical reference tools",
        "Search PubMed for treatment protocol validation and evidence",
        "Calculate medical necessity score based on evidence findings",
        "Search historical fraud databases for pattern matching",
        "Verify provider credentials against healthcare databases",
        "Analyze billing patterns using statistical tools",
        "Compare claimed costs against market rate databases",
        "Validate billing codes using standard medical coding tools",
        "Calculate comprehensive fraud risk score from all indicators",

        # Phase 3: Calculation Framework
        "Extract all numeric parameters from validated rules",
        "Create calculation dependency map for sequential processing, and display these in sequence",
        "Show calculation summary in detail"
        "Generate five distinct calculation strategies using thinker_agent",
        "Score each calculation path for accuracy and policy compliance",
        "Execute chosen calculation path with step-by-step documentation",
        "Calculations traces for all calculations in detail, in latex"
        "Verify calculations using independent validation tools",
        "Document each calculation step with policy reference",

        # Phase 4: Validation
        "Compare outputs across all team findings for consistency",
        "Cross-validate calculations against policy limits",
        "Verify medical necessity scores against approved costs",
        "Calculate confidence scores for each validation step",
        "Document all validation failures with specific details",
        "Route failed validations to respective teams for correction",
        "Re-verify all corrections using original validation criteria",

        # Phase 5: Decision Processing
        "Validate final calculations against policy terms",
        "Generate decision rationale based on validated findings",
        "Create simplified explanation of all decisions",
        "Document all tool usage and validation steps",
        "Record all resolution paths for failed validations",

        # Phase 6: Report Generation
        "Compile executive summary with confidence metrics",
        "Present medical analysis with evidence citations",
        "Document fraud risk assessment with quantified indicators",
        "Show financial calculations with validation checks",
        "Present final decision with confidence scores",
        "Generate complete audit trail of process steps",
        "Create recommendations based on process insights",

        # Quality Control
        "Verify all calculations with validation tools",
        "Confirm all evidence citations and references",
        "Validate all data point sources",
        "Check formatting consistency across report",
        "Verify completion of all required sections"
    ],
    knowledge=knowledge_base,
    read_chat_history=True,
    search_knowledge=True,
    markdown=True,
    reasoning=True
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

    # Agentic Adjudication Scrutiny Report
    Generate a comprehensive report following these specific requirements:

    ## Executive Section
    Create an executive brief containing:
    * Single-paragraph case summary
    * Claim identifiers and key dates table
    * Final decision with confidence score
    * Top 5 critical findings with detailed reasoning (Hint: Use tools).

    ## Policy Rule Analysis
    Document structured rule extraction:
    * List applicable policy sections table with citation
    * Coverage terms validation matrix
    * Exclusion applicability check
    * Waiting period validation

    ## Medical Scrutiny
    Present medical analysis with evidence:
    | Diagnosis | ICD Code | Evidence Source | Confidence |
    |-----------|----------|-----------------|------------|

    Document treatment validation:
    * Protocol alignment with standard practices
    * PubMed/medical tool citations
    * Medical necessity score

    Present documentation audit:
    | Document Type | Status | Quality Score | Gaps |
    |--------------|--------|---------------|------|

    ## Risk Evaluation
    Present quantified risk assessment:
    * Fraud score with contributing factors
    * Compliance validation metrics
    * Provider credentialing status
    * Historical pattern analysis results

    ## Financial Analysis
    Show step-wise calculation process:
    1. Base eligibility calculation
    2. Sub-limit applications
    3. Package rate adjustments
    4. Final amount determination
    5. Calculations traces for all calculations in detail

    Present expense validation:
    | Category | Claimed | Allowed | Rationale |
    |----------|---------|----------|-----------|

    Show calculation paths:
    * Path A: [Primary calculation approach]
    * Path B: [Alternative calculation approach]
    * Selected Path: [Justification with confidence score]

    ## Validation Summary
    Document validation steps:
    | Check Type | Result | Tool Used | Score |
    |------------|--------|-----------|-------|

    ## Decision Details
    Present final determination:
    * Approved amount with confidence score
    * Policy clause references
    * Validation checkpoint clearances

    ## Patient Communication
    Provide clear explanation:
    * Decision summary in simple language
    * Key factors affecting decision
    * Next steps or requirements

    ## Process Documentation
    Show complete audit trail:
    | Action | Tool/Agent | Outcome |
    |--------|------------|----------|

    ## Quality Metrics
    Report quality indicators:
    * Overall confidence score
    * Evidence strength metrics
    * Validation completion status
    * Tool utilization summary

    ## Improvement Insights
    Present actionable recommendations:
    * Process enhancement opportunities
    * Documentation improvement needs
    * Risk mitigation suggestions

    Required Standards:
    1. All calculations must show tool validation
    2. Every finding must cite evidence source
    3. Each validation must show confidence score
    4. All policy references must be specific
    5. Medical findings must have tool verification
    6. Risk scores must show calculation basis

    Format Requirements:
    1. Use clear section headings
    2. Present complex data in tables
    3. Show calculation steps clearly
    4. Include validation scores consistently
    5. Maintain evidence citations throughout
    """
    return master_team_v2.run(prompt, stream=True, stream_intermediate_steps=True)

def get_detailed_adjudication_prompt_old(claim_details):
    prompt = f"""
    Claim Details: {claim_details}

    # Agentic Adjudication Scrutiny Report

    ## Executive Summary
    Provide a concise yet comprehensive overview containing:
    - Claim Reference & Basic Details
    - Key Decision Points
    - Critical Findings
    - Final Verdict

    ## Policyholder & Claim Context
    | Parameter | Details |
    |-----------|---------|
    | Policy Number | [Value] |
    | Claim Reference | [Value] |
    | Admission Date | [Value] |
    | Discharge Date | [Value] |
    | Total Claimed Amount | [Value] |

    Provide a brief narrative of the case context and presenting condition.

    ## Medical Assessment
    ### Diagnosis & Treatment Analysis
    Provide a detailed analysis of:
    - Primary diagnosis with supporting evidence
    - Secondary conditions and their relevance
    - Treatment chronology and rationale

    ### ICD Coding Validation
    | Condition | ICD Code | Clinical Evidence |
    |-----------|----------|-------------------|

    ### Medical Necessity Evaluation
    Present a comprehensive analysis of:
    - Treatment protocol appropriateness
    - Alignment with standard medical practices
    - Evidence-based validation of procedures

    ### Documentation Assessment
    | Critical Document | Status | Quality Check |
    |-------------------|--------|---------------|

    Highlight any documentation gaps or concerns.

    ## Risk & Compliance Analysis
    ### Fraud Risk Assessment
    Provide detailed analysis of:
    - Historical patterns
    - Behavioral indicators
    - System-generated alerts

    ### Critical Risk Indicators
    | Risk Category | Finding | Evidence | Severity |
    |--------------|---------|-----------|-----------|

    ### Policy Compliance Validation
    Present narrative analysis of:
    - Waiting period adherence
    - Pre-existing condition implications
    - Policy exclusion checks

    ## Financial Scrutiny
    
     ### Rules and Calculation Result
    - Think step wise generate rules and calculations to verify adjudication risk based on policy and claim case
    - Execute all the rules and verify results.
    
    ### Expense Analysis
    | Category | Claimed | Benchmark | Variance |
    |----------|---------|-----------|-----------|

    Provide analysis of significant variances.

    ### Calculation Methodology
    Present Tree-of-Thoughts analysis:
    1. Path A: [Methodology Name]
       - Assumptions
       - Step-by-step calculation
       - Policy alignment

    2. Path B: [Methodology Name]
       - Assumptions
       - Step-by-step calculation
       - Policy alignment

    ### Selected Approach Rationale
    Explain:
    - Why the chosen path is optimal
    - Policy clause alignments
    - Risk mitigation aspects
    

    ## Final Determination
    ### Amount Summary
    | Component | Amount | Policy Reference |
    |-----------|--------|------------------|

    ### Decision Rationale
    Provide clear explanation of:
    - Key decision factors
    - Policy clause applications
    - Special considerations

    ### Patient Communication Summary
    Present in clear, non-technical language:
    - Decision explanation
    - Key points of consideration
    - Next steps if any

    ## Process Audit Trail
    Present key milestones and decisions:
    | Timestamp | Critical Action | Outcome |
    |-----------|-----------------|---------|

    ## Recommendations & Insights
    ### Process Improvements
    Outline actionable recommendations for:
    - Documentation enhancement
    - Process efficiency
    - Risk mitigation

    ### Special Notes
    Highlight any:
    - Precedent-setting aspects
    - Unique case features
    - Learning opportunities

    Quality Requirements:
    1. All assertions must be evidence-backed
    2. Policy references must be specific and accurate
    3. Calculations must show clear methodology
    4. Medical terminology must be explained
    5. Risk assessments must be quantified where possible
    6. Decisions must show clear reasoning chains

    Formatting Guidelines:
    1. Use tables only for comparative or multi-parameter data
    2. Present narrative analysis in clear, concise paragraphs
    3. Use bullet points for lists and key points
    4. Maintain consistent heading hierarchy
    5. Include cross-references where relevant
    6. Bold critical information and decisions
    
    Think Step wise and execute maximum steps for accuracy and reliability.
    """
    return master_team.run(prompt, stream=True, stream_intermediate_steps=True)


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
    return benefit_calculator.run(prompt, stream=True, stream_intermediate_steps=True)
