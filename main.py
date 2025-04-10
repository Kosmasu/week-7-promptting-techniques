import requests
import json
import time
import pandas as pd
from datetime import datetime
import os
import re
from typing import Callable, List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field, HttpUrl, validator

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODELS = ["llama3:8b", "mistral:7b", "gemma:7b"]
OUTPUT_DIR = "phishing_experiment_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Define Pydantic models
class EmailSample(BaseModel):
    id: str
    content: str
    is_phishing: bool


class OllamaRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False


class OllamaResponse(BaseModel):
    response: str


class ExperimentResult(BaseModel):
    email_id: str
    technique: str
    is_phishing: bool
    result_file: str
    timestamp: str


class PhishingAnalysisResult(BaseModel):
    email_id: str
    model_name: str
    ground_truth_is_phishing: bool
    ground_truth_content: str
    predicted_is_phishing: bool
    prompt_technique: str
    result_file: str
    timestamp: str


# Sample test emails with Pydantic models
TEST_EMAILS: List[EmailSample] = [
    # Phishing emails
    EmailSample(
        id="phish_1",
        content="""Account Security Alert: Your Facebook password needs to be changed immediately. 
        Click to reset: http://fb-security-update.xyz/reset-password
        Failure to respond within 12 hours will result in account suspension.""",
        is_phishing=True,
    ),
    EmailSample(
        id="phish_2",
        content="""Invoice #INV-98452 Payment Overdue - URGENT
        Please review the attached invoice and make payment through our secure portal:
        https://payment-portal-invoices.com/pay-now
        Your account will be charged automatically in 48 hours.""",
        is_phishing=True,
    ),
    # EmailSample(
    #     id="phish_3",
    #     content="""DHL Delivery Notification: Package held at customs
    #     Please confirm your address and payment information here:
    #     http://dhl-customs-clearance-0321.net/package?id=34921
    #     Required for delivery within 24 hours.""",
    #     is_phishing=True,
    # ),
    # EmailSample(
    #     id="phish_4",
    #     content="""IRS Tax Refund Available: You have unclaimed tax refund of $852.60
    #     Submit your banking details through our secure form:
    #     https://irs-refund-claims.org/2024/form_8911
    #     Deadline: March 15, 2024""",
    #     is_phishing=True,
    # ),
    # EmailSample(
    #     id="phish_5",
    #     content="""Job Offer: Marketing Manager Position at Amazon
    #     Click to review offer details and complete onboarding:
    #     http://amazon-careers-hiring.com/offer/accept
    #     Requires immediate confirmation and SSN verification""",
    #     is_phishing=True,
    # ),
    # Legitimate emails
    EmailSample(
        id="legit_1",
        content="""Meeting Confirmation: Project Kickoff - March 5, 2:00 PM EST
        Hi Team, Please confirm your attendance for the project kickoff meeting.
        Agenda and dial-in details attached. Best, Michael Chen""",
        is_phishing=False,
    ),
    EmailSample(
        id="legit_2",
        content="""Newsletter Subscription Confirmation
        Thank you for subscribing to The Daily Tech Digest!
        You can update your preferences anytime using your account settings.""",
        is_phishing=False,
    ),
    # EmailSample(
    #     id="legit_3",
    #     content="""Password Changed Successfully
    #     Hello Alice, Your account password was updated on March 1, 2024 at 3:15 PM PST.
    #     If you didn't make this change, contact support@trustedcompany.com immediately.""",
    #     is_phishing=False,
    # ),
    # EmailSample(
    #     id="legit_4",
    #     content="""Receipt for Purchase #789456
    #     Thank you for your order! Amount charged: $49.99
    #     Expected delivery date: March 7-10. Track your package at our official store app.""",
    #     is_phishing=False,
    # ),
    # EmailSample(
    #     id="legit_5",
    #     content="""Company All-Hands Meeting Reminder
    #     Reminder: Q1 All-Hands meeting tomorrow at 10 AM in the main conference room.
    #     Lunch will be provided. Please bring your employee badge.""",
    #     is_phishing=False,
    # ),
]


def call_ollama(
    prompt: str, model_name: str, system_prompt: Optional[str] = None
) -> str:
    """Call Ollama API with given prompt, model and optional system prompt."""
    # Default system prompt if none is provided
    if not system_prompt:
        system_prompt = "You are a helpful AI assistant."

    # Format using common template
    prompt_template = f"<|begin_of_text|><|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"

    request_data = OllamaRequest(model=model_name, prompt=prompt_template)

    try:
        response = requests.post(OLLAMA_API_URL, json=request_data.dict())
        response.raise_for_status()
        result = OllamaResponse(**response.json())
        return result.response
    except Exception as e:
        print(f"Error calling Ollama API with model {model_name}: {e}")
        return f"Error: {str(e)}"


def save_result(
    technique: str, email_id: str, model_name: str, prompt: str, response: str
) -> str:
    """Save the prompt and response to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{OUTPUT_DIR}/{technique}_{email_id}_{model_name}_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write(f"TECHNIQUE: {technique}\n")
        f.write(f"EMAIL ID: {email_id}\n")
        f.write(f"MODEL: {model_name}\n")
        f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n--- PROMPT ---\n\n")
        f.write(prompt)
        f.write(f"\n\n--- RESPONSE ---\n\n")
        f.write(response)

    print(f"Result saved to {filename}")
    return filename


def apply_automate_framework(email: EmailSample, model_name: str) -> Tuple[str, str]:
    """Apply the AUTOMATE framework technique."""
    system_prompt = """
    You are an expert cybersecurity analyst specializing in phishing detection.
    """

    prompt = f"""
    Act as an expert cybersecurity analyst with specialization in phishing detection
    User Persona & Audience: Security operations team members who need to quickly triage suspicious emails
    Targeted Action: Analyze the provided email, identify phishing indicators, and provide a risk assessment
    Output Definition: Generate a structured analysis with the following sections:
    - Phishing Verdict (Yes/No/Maybe)
    - Risk Score (1-10)
    - Identified Tactics
    - Technical Indicators
    - Recommended Actions
    Mode / Tonality / Style: Professional, technical, concise security analysis
    Atypical Cases: If the email appears to be non-English, still analyze any URLs or email addresses visible
    Topic Whitelisting: Focus only on phishing detection relevant factors from the provided email
    Eliminate Garbage Text: Ignore email signatures, legal disclaimers, and thread history markers
    
    Here is the email to analyze:
    
    "{email.content}"
    """

    response = call_ollama(prompt, model_name, system_prompt)
    return prompt, response


def apply_few_shot_learning(email: EmailSample, model_name: str) -> Tuple[str, str]:
    """Apply the few-shot learning technique."""
    system_prompt = """
    You are an expert cybersecurity analyst specializing in phishing detection.
    """

    prompt = f"""
    I want you to analyze emails for phishing attempts. Here are some examples of how to analyze them:

    EXAMPLE 1:
    Email: "Dear Customer, Your account will be suspended. Click here to verify: http://amaz0n-secure.tk"
    Analysis:
    - Phishing Verdict: Yes
    - Risk Score: 8/10
    - Identified Tactics: Urgency creation, impersonation, suspicious URL (TLD .tk)
    - Technical Indicators: Non-corporate domain, URL mismatch, generic greeting
    - Recommended Actions: Delete email, report to security team, do not click links

    EXAMPLE 2:
    Email: "Hi John, Attached is the Q3 budget report we discussed yesterday. Let me know if you need any clarification. Regards, Sarah - Finance Department"
    Analysis:
    - Phishing Verdict: No
    - Risk Score: 2/10
    - Identified Tactics: None identified
    - Technical Indicators: No suspicious links, expected attachment, personalized context, sender matches expected communication pattern
    - Recommended Actions: Normal processing, verify sender if attachment seems unusual

    Now analyze this email:
    "{email.content}"
    """

    response = call_ollama(prompt, model_name, system_prompt)
    return prompt, response


def apply_chain_of_thought(email: EmailSample, model_name: str) -> Tuple[str, str]:
    """Apply the chain of thought technique."""
    system_prompt = """
    You are an expert cybersecurity analyst specializing in phishing detection.
    """

    prompt = f"""
    As a cybersecurity expert, analyze the following email for phishing indicators. Think step by step:

    1. First, examine the sender address and compare it to the claimed identity
    2. Next, identify any urgency/pressure tactics or emotional manipulation
    3. Analyze all URLs by checking domain names and paths
    4. Look for grammatical errors, unusual phrasing, or inconsistencies
    5. Assess if the request in the email is typical or suspicious
    6. Consider the appropriate technical response
    7. Calculate an overall risk score

    Email: "{email.content}"

    Let's think through each step carefully before providing a final assessment.
    """

    response = call_ollama(prompt, model_name, system_prompt)
    return prompt, response


def apply_multi_prompt(email: EmailSample, model_name: str) -> Tuple[str, str]:
    """Apply the multi-prompt approach technique."""
    system_prompt = """
    You are an expert cybersecurity analyst specializing in phishing detection.
    """

    # First prompt - Technical analysis
    prompt1 = f"""
    Analyze the following email focusing ONLY on technical indicators of phishing:
    - Sender address analysis
    - URL/domain inspection
    - Attachment analysis
    - Header anomalies
    - Any technical deception methods

    Email: "{email.content}"
    """

    response1 = call_ollama(prompt1, model_name, system_prompt)

    # Second prompt - Social engineering analysis
    prompt2 = f"""
    Analyze the following email focusing ONLY on social engineering tactics:
    - Urgency/pressure tactics
    - Authority impersonation
    - Emotional manipulation
    - Unusual requests
    - Linguistic red flags

    Email: "{email.content}"
    """

    response2 = call_ollama(prompt2, model_name, system_prompt)

    # Third prompt - Integration
    prompt3 = f"""
    Based on the following technical and social engineering analyses, provide a unified phishing risk assessment:

    Technical Analysis: {response1}
    
    Social Engineering Analysis: {response2}

    Include:
    - Final phishing verdict
    - Risk score (1-10)
    - Recommended actions
    """

    response3 = call_ollama(prompt3, model_name, system_prompt)

    # Combine all prompts and responses for saving
    combined_prompt = f"PROMPT 1 (Technical):\n{prompt1}\n\nPROMPT 2 (Social):\n{prompt2}\n\nPROMPT 3 (Integration):\n{prompt3}"
    combined_response = f"RESPONSE 1 (Technical):\n{response1}\n\nRESPONSE 2 (Social):\n{response2}\n\nRESPONSE 3 (Integration):\n{response3}"

    return combined_prompt, combined_response


def apply_two_step_prompting(email: EmailSample, model_name: str) -> Tuple[str, str]:
    """Apply the two-step prompting technique."""
    system_prompt = """
    You are an expert cybersecurity analyst specializing in phishing detection.
    """

    # Step 1 - Detailed reasoning
    prompt1 = f"""
    You are an expert cybersecurity analyst. I want you to analyze this email for phishing indicators. Think through all possible signs of legitimacy or deception. Consider technical indicators, social engineering tactics, and contextual anomalies. Document your complete reasoning process.

    Email: "{email.content}"
    """

    response1 = call_ollama(prompt1, model_name, system_prompt)

    # Step 2 - Format output
    prompt2 = f"""
    Based on your detailed analysis, format your findings into a concise security report with the following sections:
    - Phishing Verdict (Yes/No/Maybe)
    - Risk Score (1-10)
    - Key Indicators (bullet points)
    - Recommended Actions (bullet points)

    Your analysis: {response1}
    """

    response2 = call_ollama(prompt2, model_name, system_prompt)

    # Combine both steps for saving
    combined_prompt = (
        f"PROMPT 1 (Reasoning):\n{prompt1}\n\nPROMPT 2 (Formatting):\n{prompt2}"
    )
    combined_response = (
        f"RESPONSE 1 (Reasoning):\n{response1}\n\nRESPONSE 2 (Formatting):\n{response2}"
    )

    return combined_prompt, combined_response


def parse_phishing_verdict(response: str) -> bool:
    """
    Parse the LLM response to determine the phishing verdict (Yes/No/Maybe).
    Returns True for "Yes", False for "No", and makes a best guess for "Maybe" based on risk score.
    """
    # First look for a direct verdict
    verdict_patterns = [
        r"Phishing Verdict:?\s*(Yes|No|Maybe)",
        r"Final phishing verdict:?\s*(Yes|No|Maybe)",
        r"Verdict:?\s*(Yes|No|Maybe)",
        r"Phishing:?\s*(Yes|No|Maybe)",
        r"is (?:a|this)? phishing (?:email|attempt)?:?\s*(Yes|No|Maybe)",
    ]

    for pattern in verdict_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            verdict = match.group(1).lower()
            if verdict == "yes":
                return True
            elif verdict == "no":
                return False
            # For "maybe", we'll check the risk score below

    # If no direct verdict, check risk score
    risk_patterns = [r"Risk Score:?\s*(\d+)(?:/10)?", r"Risk:?\s*(\d+)(?:/10)?"]

    for pattern in risk_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            risk_score = int(match.group(1))
            # Consider risk scores of 5 or higher as phishing
            return risk_score >= 5

    # If all else fails, look for keywords
    phishing_indicators = [
        "suspicious",
        "malicious",
        "phishing",
        "scam",
        "fraud",
        "fake",
        "deceptive",
    ]
    for indicator in phishing_indicators:
        if re.search(rf"\b{indicator}\b", response, re.IGNORECASE):
            return True

    # Default to False if nothing conclusive found
    return False


def create_phishing_analysis_result(
    email: EmailSample, technique: str, model_name: str, response: str, filename: str
) -> PhishingAnalysisResult:
    """Create a PhishingAnalysisResult object by parsing the LLM response."""
    predicted_is_phishing = parse_phishing_verdict(response)

    return PhishingAnalysisResult(
        email_id=email.id,
        model_name=model_name,
        ground_truth_is_phishing=email.is_phishing,
        ground_truth_content=email.content,
        predicted_is_phishing=predicted_is_phishing,
        prompt_technique=technique,
        result_file=filename,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

def unload_model(model_name: str) -> bool:
    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "keep_alive": 0}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Model '{model_name}' unloadeded successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to unload model '{model_name}': {e}")
        return False

def run_experiment() -> List[PhishingAnalysisResult]:
    """Run the full experiment with all techniques, models on all test emails."""
    results: List[PhishingAnalysisResult] = []

    techniques: List[Tuple[str, Callable]] = [
        ("AUTOMATE", apply_automate_framework),
        ("Few-Shot", apply_few_shot_learning),
        ("Chain-of-Thought", apply_chain_of_thought),
        ("Multi-Prompt", apply_multi_prompt),
        ("Two-Step", apply_two_step_prompting),
    ]

    total_combinations = len(TEST_EMAILS) * len(techniques) * len(MODELS)
    print(
        f"Starting experiment with {len(TEST_EMAILS)} test emails, {len(techniques)} techniques, and {len(MODELS)} models ({total_combinations} total combinations)"
    )

    for model_name in MODELS:
        for email in TEST_EMAILS:
            for technique_name, technique_func in techniques:
                print(
                    f"Applying {technique_name} with {model_name} to email {email.id}..."
                )

                try:
                    # Update the function call to include model_name
                    prompt, response = technique_func(email, model_name)
                    filename = save_result(
                        technique_name, email.id, model_name, prompt, response
                    )

                    # Parse the response and create a PhishingAnalysisResult
                    analysis_result = create_phishing_analysis_result(
                        email=email,
                        technique=technique_name,
                        model_name=model_name,
                        response=response,
                        filename=filename,
                    )

                    results.append(analysis_result)

                except Exception as e:
                    print(
                        f"Error applying {technique_name} with {model_name} to {email.id}: {e}"
                    )
        unload_model(model_name)

    # Save detailed results to CSV
    results_df = pd.DataFrame([result.dict() for result in results])
    results_df.to_csv(f"{OUTPUT_DIR}/phishing_analysis_results.csv", index=False)
    print(
        f"Experiment completed. Detailed results saved to {OUTPUT_DIR}/phishing_analysis_results.csv"
    )

    return results


def analyze_results(results: List[PhishingAnalysisResult]) -> None:
    """Perform detailed analysis of results after experiment."""
    results_df = pd.DataFrame([result.dict() for result in results])

    print("\n=== Analysis Results ===")

    # 1. Overall accuracy by model
    print("\n--- Overall Accuracy by Model ---")
    model_accuracy = (
        results_df.groupby("model_name")
        .apply(
            lambda x: (
                x["predicted_is_phishing"] == x["ground_truth_is_phishing"]
            ).mean()
        )
        .reset_index(name="accuracy")
    )
    print(model_accuracy)

    # 2. Accuracy by model and technique
    print("\n--- Accuracy by Model and Technique ---")
    model_technique_accuracy = (
        results_df.groupby(["model_name", "prompt_technique"])
        .apply(
            lambda x: (
                x["predicted_is_phishing"] == x["ground_truth_is_phishing"]
            ).mean()
        )
        .reset_index(name="accuracy")
    )
    print(model_technique_accuracy)

    # 3. False positive and false negative rates by model
    print("\n--- Error Rates by Model ---")

    def calculate_error_rates(group):
        true_phishing = group[group["ground_truth_is_phishing"] == True]
        true_legitimate = group[group["ground_truth_is_phishing"] == False]

        false_negative_rate = 0
        if len(true_phishing) > 0:
            false_negative_rate = (
                true_phishing["predicted_is_phishing"] == False
            ).mean()

        false_positive_rate = 0
        if len(true_legitimate) > 0:
            false_positive_rate = (
                true_legitimate["predicted_is_phishing"] == True
            ).mean()

        return pd.Series(
            {
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
            }
        )

    error_rates = (
        results_df.groupby("model_name").apply(calculate_error_rates).reset_index()
    )
    print(error_rates)

    # 4. Best performing technique for each model
    print("\n--- Best Technique for Each Model ---")
    best_techniques = model_technique_accuracy.sort_values(
        ["model_name", "accuracy"], ascending=[True, False]
    )
    best_techniques = best_techniques.groupby("model_name").first().reset_index()
    print(best_techniques[["model_name", "prompt_technique", "accuracy"]])

    # 5. Create visualizations
    create_visualizations(
        results_df, model_accuracy, model_technique_accuracy, error_rates
    )

    print("\nAnalysis completed. Visualizations saved to the output directory.")


def create_visualizations(
    results_df, model_accuracy, model_technique_accuracy, error_rates
):
    """Create and save visualizations for the experiment results."""
    # Ensure matplotlib is available
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # 1. Overall model accuracy comparison
    plt.figure(figsize=(10, 6))
    accuracy_bar = sns.barplot(x="model_name", y="accuracy", data=model_accuracy)
    plt.title("Overall Accuracy by Model")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    for i, v in enumerate(model_accuracy["accuracy"]):
        accuracy_bar.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_accuracy_comparison.png")
    plt.close()

    # 2. Accuracy by model and technique
    plt.figure(figsize=(12, 8))
    technique_comparison = sns.barplot(
        x="model_name",
        y="accuracy",
        hue="prompt_technique",
        data=model_technique_accuracy,
    )
    plt.title("Accuracy by Model and Technique")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.legend(title="Technique", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/technique_comparison.png")
    plt.close()

    # 3. Error rates comparison
    plt.figure(figsize=(12, 6))
    melted_errors = pd.melt(
        error_rates,
        id_vars=["model_name"],
        value_vars=["false_positive_rate", "false_negative_rate"],
        var_name="error_type",
        value_name="rate",
    )
    error_comparison = sns.barplot(
        x="model_name", y="rate", hue="error_type", data=melted_errors
    )
    plt.title("Error Rates by Model")
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.xlabel("Model")
    plt.legend(title="Error Type")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/error_rates.png")
    plt.close()

    # 4. Heatmap of model-technique performance
    plt.figure(figsize=(12, 8))
    pivot_data = model_technique_accuracy.pivot(
        index="prompt_technique", columns="model_name", values="accuracy"
    )
    heatmap = sns.heatmap(
        pivot_data, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f"
    )
    plt.title("Model-Technique Performance Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap.png")
    plt.close()


if __name__ == "__main__":
    print("=== Phishing Detection Prompt Engineering Experiment ===")
    print(f"Using models: {', '.join(MODELS)}")
    
    # Check if Ollama is running
    try:
        test_response = requests.get("http://localhost:11434/api/version")
        if test_response.status_code == 200:
            print("Ollama is running. Starting experiment...")
            results = run_experiment()
            analyze_results(results)
        else:
            print("Ollama seems to be running but returned an unexpected response.")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please make sure Ollama is installed and running on http://localhost:11434")