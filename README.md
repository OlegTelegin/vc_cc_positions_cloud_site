# vc_cc_positions_cloud_site

Single-page static site for `positiontitles.olegteleg.in`.

## Deployment (GitHub Pages)

This repository is configured for GitHub Pages from the branch root:

- Root `index.html` redirects to `web/`
- App files live in `web/`
- `CNAME` pins custom domain to `positiontitles.olegteleg.in`
- `.nojekyll` disables Jekyll processing

## Your required settings

1. GitHub -> Repository -> Settings -> Pages
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/ (root)`
2. Custom domain: `positiontitles.olegteleg.in`
3. Enable `Enforce HTTPS` (once certificate is issued)

## DNS checks (GoDaddy)

For subdomain `positiontitles.olegteleg.in`, create:

- Type: `CNAME`
- Host/Name: `positiontitles`
- Value/Points to: `<your-github-username>.github.io`
- TTL: default is fine

Also make sure there is **no conflicting A/AAAA/CNAME record** for the same host `positiontitles`.

## Verification

After pushing to `main`, wait a few minutes, then verify:

- `https://positiontitles.olegteleg.in` opens the site
- `https://positiontitles.olegteleg.in/web/` opens directly
- GitHub Pages status shows: `Your site is live`