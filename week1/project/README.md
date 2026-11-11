# 🧱 Week 1 project: the exam pack → JSON (v1)

This is your first GradeWise milestone, and it's a big one! 🎉 Asha hands you three PDFs for the CS201 midterm: the **question paper**, the **marking scheme** and the **marking methodology**. By the end of Day 5, GradeWise turns them into **one validated `ExamPaper`**: a JSON array of questions, each carrying its own marking criteria and the examiner's method. Every later week grades against this file.

```bash
uv run week1/project/extract_exam.py
```

You'll see a table of the questions, then a green panel, and the result lands in `output/cs_midterm.json`.

## 🗺️ How it works (one call, no magic)
1. **Read** each PDF as Markdown with `pymupdf4llm` (headings, bold and tables survive, so the model reads them well).
2. **Prompt** the model with all three documents under clear headings, plus a short checklist of rules: stable IDs like `Q7a` and `Q7a.c1`, copy the question text exactly, and **never invent marks**.
3. **Structured output:** `chat_model().with_structured_output(ExamPaper)` makes the model answer in the exact shape of the schema in `gradewise/schemas/exam.py`. Pydantic checks the shape for you.
4. **Check the arithmetic** with `paper.problems()`: do the criteria add up to each question's marks, and the questions to the paper's total?

## 🧪 Things to try
- Delete rule 4 ("NEVER invent marks") from `SYSTEM_PROMPT` and run again. Do the totals still add up?
- Set `CHAT_MODEL` to a smaller or larger model in `.env` and compare the JSON.
- Open `output/cs_midterm.json` next to `data/exams/cs_midterm/expected_exam.json`. What did the model get differently?

## 🆘 If something goes wrong
- **Red "Missing API key" panel:** copy `.env.example` to `.env`, paste your key, and *save the file*.
- **"Course data not found":** run `git pull`; the exam pack lives in `data/exams/cs_midterm/`.
- **Warnings that marks don't add up:** that's exactly the problem v1 can't fix on its own. In **week 3** we rebuild this as a LangGraph workflow that sends the problems back to the model, retries, and asks Asha to approve before saving (`week3/project/`).

⏭ The packaged version of this code lives in `gradewise/exam/` (`extract_exam`, `extract_with_retry`).
