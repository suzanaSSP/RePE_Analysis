## Phase 0: Setup

- **Pick models:** Use one instruction-tuned model with good tooling, like Llama 3.1 8B Instruct or Gemma 2 9B-it. Gemma has the advantage of Gemma Scope SAEs if you want to dig into features later. If time allows, add a second model so your results don't depend on one model's quirks.
- **Set up tooling:** TransformerLens or nnsight both work for caching activations and adding steering hooks. Always run prompts through the model's chat template, because refusal behavior changes a lot without it. **Actually, I'll be using this github from the paper: https://github.com/andyzoujm/representation-engineering for vector sterring**
- **Write down your hypotheses:** Do this before running anything. For example: "Religiosity steering increases refusal on contested and benign prompts more than on clearly harmful ones." Pre-stating this keeps you honest when results come in.
**Hypothesis: Religiosity steering increases refusal clearly harmful and unethical prompts, and on contested prompts.**

## Phase 1: Build datasets

-  ✅ **Contrastive pairs for the religiosity direction:** Aim for roughly 200–500 pairs. Each pair is a religious text and a neutral version matched on topic, length, and formality. Draw the religious side from several traditions so no single one dominates. Hold out 20% for validation.
-  ✅ **Control pairs:** Build a similar set contrasting formal, morally framed secular text with neutral text. This gives you a "moral tone" vector to compare against.
- **Evaluation prompt sets:** You need three:
  - **Harmful:** a subset of HarmBench or SORRY-Bench.
  - **Ethically contested:** you'll likely need to write these yourself, about 100 prompts on topics like alcohol, gambling, or lifestyle choices where religious and secular norms tend to differ. Have a second person check your labels.
  - **Benign:** XSTest or OR-Bench, which include harmless prompts that look risky, for measuring over-refusal.

## Phase 2: Extract directions

- **Cache activations:** Run your contrastive pairs and record the residual stream at the final token for every layer.
- **Compute the religiosity direction:** At each layer, take the mean activation of religious texts minus the mean of neutral texts.
- **Pick a layer:** Train a linear probe at each layer on the training pairs and test it on the held-out ones. Choose a layer where accuracy is high, usually somewhere in the middle of the network.
- **Extract the refusal direction:** Extract it the same way Arditi et al. (2024) do: mean activation on harmful prompts minus mean on harmless prompts. Their code is a good template.
- **Measure overlap:** Compute the cosine similarity between the religiosity, refusal, and control directions. Record this early, because it shapes how you interpret everything else.

## Experiment 1: Validate the vector

- Steer about 50 neutral prompts at several strengths (for example 0, 2, 4, 8, scaled relative to the typical size of activations at that layer).
- Score each output's religiosity with an LLM judge, and track perplexity or a fluency score. Spot-check some outputs yourself.
- Check balance across traditions: do steered outputs default to one tradition's vocabulary?
- **Result:** a working range of strengths where outputs are clearly more religious but still coherent. Only use strengths in this range from here on.

## Experiment 2: Behavioral refusal

- Run all three prompt sets under four conditions: no steering, religiosity steering, control (moral tone) steering, and a random vector of the same size.
- Classify each response as refuse or comply. A string-matching first pass ("I can't", "I won't") plus an LLM judge for ambiguous cases works well.
- Plot refusal rate against steering strength for each prompt category. Use bootstrap confidence intervals, and use McNemar's test for paired comparisons on the same prompts.
- **Key question:** does religiosity raise refusal more than the moral-tone control and the random vector, and in which categories?

## Experiment 3: Internal speed

- For each prompt, project the residual stream onto the refusal direction at every layer, with and without religiosity steering.
- Find the earliest layer where that projection crosses a threshold. You can also use a logit or tuned lens to see when refusal tokens like "I" in "I can't" become likely. Your lens experience transfers directly here.
- **Key question:** does steering make refusal appear at earlier layers? This is the most direct test of the "how quickly" part of your question.

## Experiment 4: Disentangle the mechanism

- **Orthogonalize:** remove the refusal-direction component from the religiosity vector, then rerun Experiment 2. If the effect disappears, religiosity works by overlapping with refusal. If it remains, it works through some other route.
- **Ablate:** remove the refusal direction from the model while steering with religiosity. If refusal still rises, something besides the known refusal direction is responsible.

## Optional extensions

- **Output speed:** count how many tokens the model generates before refusing, and whether it ever starts complying and then backs out.
- **Base vs. instruct:** run Experiment 2 on the base model to avoid ceiling effects on harmful prompts, since instruct models already refuse nearly all of them.

## Common pitfalls

- Steering too strongly breaks the model, and a broken model's output can look like refusal. That's why Experiment 1's working range matters.
- Results can change depending on whether you steer at every token position or only the prompt tokens. Choose one, and say which in your write-up.
- A refusal increase on benign prompts is over-refusal, not better safety. Report it that way.

***

*A reasonable pace is one phase or experiment every week or two. Experiments 1 and 2 alone make a solid result; 3 and 4 are what turn it into a mechanistic paper.*
