# 🚀 GitHub Pages Setup for Coverage Reports

## Overview

Coverage reports are automatically generated and committed to `docs/coverage/` on every push to the `develop` branch. They are served alongside your official documentation via GitHub Pages.

## Quick Setup

### Step 1: Enable GitHub Pages

1. Go to: https://github.com/fulgidus/heimdall/settings/pages
2. Under **"Build and deployment":**
   - **Branch:** Select `develop`
   - **Folder:** Select `/docs`
3. Click **Save**

**Result:** Your site will be live at https://fulgidus.github.io/heimdall

### Step 2: Access Coverage Reports

After enabling Pages, coverage reports are available at:

```
📊 https://fulgidus.github.io/heimdall/coverage/
├── index.html                    ← Dashboard
├── badge.svg                     ← Coverage badge
├── README.md                     ← Summary
├── backend_latest/               ← Backend coverage HTML
│   ├── index.html
│   ├── htmlcov/
│   └── ...
└── frontend_latest/              ← Frontend coverage HTML
    ├── index.html
    └── ...
```

## Accessing Reports

| Report | URL |
|--------|-----|
| **Dashboard** | https://fulgidus.github.io/heimdall/coverage/ |
| **Backend Report** | https://fulgidus.github.io/heimdall/coverage/backend_latest/ |
| **Frontend Report** | https://fulgidus.github.io/heimdall/coverage/frontend_latest/ |
| **Badge SVG** | https://fulgidus.github.io/heimdall/coverage/badge.svg |

## Workflow Behavior

The [coverage.yml workflow](.github/workflows/coverage.yml):

1. ✅ Runs pytest and vitest on push to `develop`
2. ✅ Generates HTML reports and SVG badge
3. ✅ Copies reports to `docs/coverage/`
4. ✅ Commits changes to `develop` branch
5. ✅ GitHub Pages serves from `develop:/docs`

## Using the Badge

Add to your `README.md`:

```markdown
[![Coverage Badge](docs/coverage/badge.svg)](docs/coverage/index.html)
```

## Troubleshooting

### Pages not loading (404 error)

**Check:**
1. Go to https://github.com/fulgidus/heimdall/settings/pages
2. Verify: **Branch** = `develop`, **Folder** = `/docs`
3. Wait 1-2 minutes for GitHub to rebuild

### Reports not updating

**Check:**
1. Push a commit to `develop`
2. Go to **Actions** tab → **Test Coverage Report**
3. Verify workflow ran successfully
4. Verify `docs/coverage/` files were committed

## Local Testing

To test coverage reports locally:

```bash
# Run backend coverage
pytest --cov=services --cov-report=html:docs/coverage/backend_latest

# Run frontend coverage
cd frontend && pnpm run test:coverage

# View locally (requires Python)
cd docs/coverage
python -m http.server 8000
# Open http://localhost:8000
```

---

**All set!** Coverage reports are automatically generated and published on every push. 🎉
