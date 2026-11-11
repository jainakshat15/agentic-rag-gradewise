# 🤝 Contributing

Built something on top of GradeWise? A new subject, a different language, a smarter grader, a slicker UI? I'd love to see it, and so would the other students.

## How to share your work

1. **Fork** this repo on GitHub and clone your fork.
2. Create a folder for yourself: `community_contributions/<your-github-username>/`
3. Put your work **only in that folder**. Please don't change files elsewhere (unless you're fixing a mistake in the course: thank you!).
4. **Clear notebook outputs** before committing. In Cursor/VS Code: the `…` menu at the top of the notebook → **Clear All Outputs**. Or install the automatic stripper once:
   ```bash
   uv run nbstripout --install
   ```
5. Add a short `README.md` in your folder: what you built, how to run it, which week it extends, and a screenshot if you can.
6. **Never commit your `.env` or API keys.** (`.gitignore` protects you, but double-check!)
7. Open a Pull Request with a title like `Week 5: Moderation second-grader by @yourname`.

## Fixing a mistake in the course
Spotted a bug or typo in a notebook or guide? PRs are very welcome. Please explain what was wrong and how you tested the fix (Windows / Mac / Linux).

## Ground rules
- Be kind and constructive in reviews and discussions.
- Use only data you have the right to share. **No real student data**, ever.
- By contributing, you agree your contribution is shared under this repo's licence.

New to Git and Pull Requests? See [guides/03_git_and_github.ipynb](guides/03_git_and_github.ipynb).
