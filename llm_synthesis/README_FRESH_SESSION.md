# Instructions for the FRESH (uncontaminated) session run

Goal: get weights from a Claude session that has seen NOTHING about this project (no memory, no project files, no RESULTS.md).

## Setup (once)
1. `unzip` the package, `pip install numpy jax` (CPU is enough). Run `python freeze.py --check` -> must print `package unchanged`. Record `package_sha256` from `FREEZE.json`.
2. Decide which rows of `prompts/run_sheet.csv` to run (priority 1 first = 12 chats; priority 2 = 24; priority 3 = 12; priority 4 = 12).

## For every row of the run sheet
1. Open a NEW chat. Memory OFF, no project, no attachments, no custom instructions. Note the model name/version shown in the UI, and whether the chat has code execution.
2. Paste the content of the row's `prompt_file` **verbatim** as the first message. Do not add anything else.
3. Save the whole conversation as `records/transcripts/<attempt_id>.md`.
4. Extract the final deliverable **without editing it**: category 3b -> `submissions/<attempt_id>.py` ; category 4 -> `submissions/<attempt_id>.json`. If it is syntactically broken or truncated, save it as is (it will count as an invalid submission; do not fix it).
5. For `feedback = none`: that is the end of the attempt.
   For `feedback = clean_check`: run `python harness/feedback.py --task T --submission S --condition C` and paste its output as your next message (max 3 revisions; after each revision save the new deliverable as `<attempt_id>_rev1.py` etc. and the final one as `<attempt_id>.py`). Stop when the output says `development-set accuracy: 200/200` or after 3 revisions.
6. Evaluate: `python harness/run_eval.py --task T --submission S --condition C --meta meta.json --out results/<attempt_id>.json` (a few minutes for correct models; use `--quick` if short of time; invalid or incorrect submissions return quickly).
7. Fill `records/<attempt_id>.json` from `records/record_template.json`.

`meta.json` example: `{"llm_identity":"<name>","llm_version":"<version>","interface":"claude.ai web","sampling_settings":"chat default, unknown","code_execution_available":"yes/no","run":"fresh","package_sha256":"<from FREEZE.json>"}`.

## Return
Zip of `submissions/`, `records/`, `results/` (and anything unusual you noticed). I will re-run every submission through the frozen harness myself, so your `results/` files are a cross-check, not the source of truth.

## Rules for you
Do not tell the LLM anything beyond the prompt. Do not retry a chat because the answer looks bad (a new chat is a new attempt; record how many chats you used per cell). Do not give feedback in `none` prompts.
