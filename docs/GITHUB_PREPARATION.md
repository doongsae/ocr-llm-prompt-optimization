# GitHub Preparation

## Decisions Before Publication

1. Select a repository name and owner, and choose public or private visibility.
2. Select a license for the original project code. None is assigned by this copy.
3. Review prompt/template attribution and dataset/weight redistribution terms.
4. Decide whether the historical `archive/` should be uploaded or stay local.
5. Review the staged file list and the publication-check report.

Dataset images, model weights, API caches, credentials, raw prediction rows,
and original third-party repositories are deliberately excluded. Aggregated
tables in `reference_results/` are historical records, not rerun results.
The archive contains obsolete methods and machine-specific paths. Do not use it
as the current method, run its tests as the supported suite, or equate historical
test-selected variants with a clean validation-selected baseline.

## Offline Checks

```bash
python tools/check_publication.py
python -m unittest discover -s pipeline/tests -v
```

The optional full data audit needs the private frozen manifest. Neither command
is permission to run paid inference.

## First Commit and Upload

Use the commands below for the first upload after confirming the repository
name, visibility, and included files. Replace the remote with the actual
repository URL; a GitHub profile URL alone is not a push destination.

```bash
git init -b main
git add README.md config.example.json .gitignore requirements.txt pipeline analysis ocr docs tools reference_results
git status --short
git diff --cached --stat
git -c core.whitespace=-blank-at-eof,cr-at-eol diff --cached --check
python tools/check_publication.py --staged
git commit -m "Add OCR prompt-optimization research snapshot"
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

Both `.gitignore` and the explicit `git add` list exclude the credential-bearing
local `config.json` and historical `archive/`. If you decide to publish the
archive, review the history and
upstream licenses, then remove that ignore entry and stage it separately. Do not use
`git add -f` for ignored secrets or generated outputs. Git LFS is not a substitute
for dataset redistribution permission or removing an exposed API key.
