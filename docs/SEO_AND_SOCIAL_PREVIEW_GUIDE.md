# SEO, social previews and verified screenshot policy

Public pages have route-specific canonical, title, description, Open Graph/Twitter image and JSON-LD WebPage metadata. The social images below are cropped from actual deployed app screenshots, with source hashes verified against existing 14 October 2026 screenshot manifests. The images are not model-performance evidence.

| Public URL | Social JPEG | Verified source PNG | Source SHA-256 | File size |
| --- | --- | --- | --- | ---: |
| https://visual-qc-project-pearl.vercel.app/ | /line/static/seo/visual-qc-social.jpg | docs/assets/visual-qc-dashboard.png | 004c3bcbf8dd9063fcbe495834ce866d478663309c35e47f1e1cecf08a72ca5e | 130788 B |
| https://visual-qc-project-pearl.vercel.app/case-study | /line/static/seo/visual-qc-social.jpg | docs/assets/visual-qc-dashboard.png | 004c3bcbf8dd9063fcbe495834ce866d478663309c35e47f1e1cecf08a72ca5e | 130788 B |
| https://visual-qc-project-pearl.vercel.app/rag | /line/static/seo/rag-social.jpg | docs/assets/visual-qc-rag.png | da29694cda3e2e5f1ed0140cb76d7d02fb68c7433cf22ebc3267c9daf5cb6085 | 109462 B |

## Interpretation and maintenance

- Fraud Analytics displays historical governed BigQuery reports, not continuously observed transactions.
- Smart Factory and Visual QC operator dashboards are simulations, not actual equipment feeds.
- A model benchmark screenshot must not be interpreted as a deployment or plant-performance result.
- Cropping and JPEG compression are image-delivery changes; no metrics or visual content were invented.
- Robots/sitemap endpoints advertise public page URLs only. Robots policy is not API authorization.
- External search indexing and social card cache refresh times are controlled by third-party crawlers.
- Regenerate preview assets only from provenance-verified screenshots and re-run the local validation script.

## Verification

Run: python scripts/check_seo_assets.py
Then verify public URLs plus /robots.txt and /sitemap.xml after release.
