# My plan
Does AI act more ethically when it has a more religious directions in activations? Does it refuse more harmful prompt?

## Hypothesis
Religiosity steering increases refusal clearly harmful and unethical prompts, and on contested prompts.

## Phase 0: Setup
    - Pick model: I'll use a Gemma model so I can do features later on
    - Set up Tooling

## Phase 1: Build Datasets
    - Contrastive pairs for the religiosity direction. Aim for roughly 200–500 pairs. Each pair is a religious text and a neutral version matched on topic, length, and formality. Draw the religious side from several traditions so no single one dominates. Hold out 20% for validation.
    - Control pairs. Build a similar set contrasting formal, morally framed secular text with neutral text. This gives you a "moral tone" vector to compare against.