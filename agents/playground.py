from phi.playground import Playground, serve_playground_app

from agents.adjudication_workflow import master_team, rules_team, medical_team, fraud_team, financial_team, \
    calculation_agent, calculation_validator, discrepancy_analyzer, review_team, explainability_agent, \
    audit_trail_agent, thinker_agent, grader_agent

app = Playground(agents=[
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
    ]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)