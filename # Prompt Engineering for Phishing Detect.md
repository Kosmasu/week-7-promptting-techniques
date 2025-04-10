# Prompt Engineering for Phishing Detection with LLMs

## 1. Introduction

This report covers my experiments testing different ways to instruct Large Language Models (LLMs) to detect phishing emails. Phishing remains one of the most common cyber attacks, and using AI could help identify suspicious messages more efficiently.

## 2. Setup and Approach

### 2.1 The Models I Used

I tested three different AI models through Ollama:

- Llama 3 (8B)
- Mistral (7B)
- Gemma (7B)

For all models, I used their standard prompt format, which for Llama 3 looks like:

```
<|begin_of_text|><|system|>
{{system_prompt}}
<|user|>
{{user_prompt}}
<|assistant|>

```

### 2.2 My Goal

I wanted to find out which instruction method works best to help AI:

- Correctly identify phishing emails
- Analyze the tricks used in the phishing attempt
- Rate how dangerous the email is (on a scale of 1-10)
- Suggest what to do about it

### 2.3 Test Emails

I created a small test set of 4 sample emails:

- 2 phishing emails with different tactics
- 2 legitimate business emails

### 2.4 Problems

Turns out running three models at the same time is too much for my computer. I even got a blue screen twice 😭😭

![WhatsApp Image 2025-04-10 at 14.08.30_3dc2d81b.jpg](attachment:55a01b4e-99c2-49ba-bf4f-b593b526a678:WhatsApp_Image_2025-04-10_at_14.08.30_3dc2d81b.jpg)

![WhatsApp Image 2025-04-10 at 14.08.31_d90cc9af.jpg](attachment:a31e4fb9-4981-4d74-9e69-0eac196dcb3b:WhatsApp_Image_2025-04-10_at_14.08.31_d90cc9af.jpg)

To get around this issue, I loaded only one model at a time by unloading the model after experimenting with it. I unloaded the model by using the `keep_alive` parameter provided by `ollama` and setting it to `0`.

```python
    for model_name in MODELS:
        for email in TEST_EMAILS:
            for technique_name, technique_func in techniques:
			          # Process here ...
        unload_model(model_name)
        
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
```

## 3. Five Different Instruction Methods

### 3.1 The AUTOMATE Framework

This method provides clear structure for the AI:

```
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

```

Results: This method gave clear, structured answers. All models followed the format well, but sometimes missed subtle phishing clues.

### 3.2 Few-Shot Learning (Teaching by Example)

I showed the AI examples of both phishing and legitimate emails with correct analyses:

```
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

Now analyze this email: [test email text]

```

Results: This approach significantly improved detection of cleverly disguised phishing attempts. The models learned from the examples and applied similar reasoning to new emails.

### 3.3 Chain of Thought (Step-by-Step Thinking)

I asked the AI to think through its analysis one step at a time:

```
As a cybersecurity expert, analyze the following email for phishing indicators. Think step by step:

1. First, examine the sender address and compare it to the claimed identity
2. Next, identify any urgency/pressure tactics or emotional manipulation
3. Analyze all URLs by checking domain names and paths
4. Look for grammatical errors, unusual phrasing, or inconsistencies
5. Assess if the request in the email is typical or suspicious
6. Consider the appropriate technical response
7. Calculate an overall risk score

Email: [test email text]

```

Results: This method produced the most detailed analysis. By following a logical sequence, the AI was less likely to miss important clues, especially social engineering tactics.

### 3.4 Multi-Prompt Approach (Breaking It Down)

I split the analysis into three separate questions:

**First Question - Technical Analysis:**

```
Analyze the following email focusing ONLY on technical indicators of phishing:
- Sender address analysis
- URL/domain inspection
- Attachment analysis
- Header anomalies
- Any technical deception methods

Email: [test email text]

```

**Second Question - Social Engineering Analysis:**

```
Analyze the following email focusing ONLY on social engineering tactics:
- Urgency/pressure tactics
- Authority impersonation
- Emotional manipulation
- Unusual requests
- Linguistic red flags

Email: [test email text]

```

**Third Question - Final Assessment:**

```
Based on the following technical and social engineering analyses, provide a unified phishing risk assessment:

Technical Analysis: [output from prompt 1]
Social Engineering Analysis: [output from prompt 2]

Include:
- Final phishing verdict
- Risk score (1-10)
- Recommended actions

```

Results: This approach provided the most thorough analysis by allowing the AI to focus on different aspects separately. However, it was also the most time-consuming, needing three separate interactions.

### 3.5 Two-Step Prompting (Think First, Then Format)

I used a two-part approach:

**Step 1 - Deep Analysis:**

```
You are an expert cybersecurity analyst. I want you to analyze this email for phishing indicators. Think through all possible signs of legitimacy or deception. Consider technical indicators, social engineering tactics, and contextual anomalies. Document your complete reasoning process.

Email: [test email text]

```

**Step 2 - Clean Report:**

```
Based on your detailed analysis, format your findings into a concise security report with the following sections:
- Phishing Verdict (Yes/No/Maybe)
- Risk Score (1-10)
- Key Indicators (bullet points)
- Recommended Actions (bullet points)

Your analysis: [output from step 1]

```

Results: This method combined thorough reasoning with a clean, structured output format. The first step allowed the AI to explore all possibilities, while the second step created an easy-to-read report.

## 4. Results and Comparison



### Key Findings:


## 5. Challenges and Limitations

## 6. Conclusion and Future Work


## 7. References

1. Ollama Documentation. (2024). Retrieved from https://ollama.com/docs
2. White, T. (2023). Prompt Engineering: A Practical Guide. O'Reilly Media.
3. The Perfect Prompt. (2024). Medium. Retrieved from https://medium.com/the-generator/the-perfect-prompt-prompt-engineering-cheat-sheet
4. PromptingGuide.ai. (2024). A Comprehensive Guide to Prompt Engineering. Retrieved from https://www.promptingguide.ai/
5. Lin, R. (2024). Applying LLMs in Cybersecurity Systems [Lecture slides].