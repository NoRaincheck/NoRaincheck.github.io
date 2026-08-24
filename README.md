# NoRaincheck

A personal blog built with [Hugo](https://gohugo.io/) and the
[hugo-bearblog](https://themes.gohugo.io/themes/hugo-bearblog/) theme,
deployed to GitHub Pages at <https://noraincheck.github.io/>.

## Development

```sh
git clone --recurse-submodules https://github.com/NoRaincheck/NoRaincheck.github.io.git
cd NoRaincheck.github.io

hugo server -D   # dev server at http://localhost:1313
hugo             # build to public/
```

Hugo (extended) >= 0.146 is required. The theme is vendored as a git
submodule under `themes/hugo-bearblog`.

## Structure

- `hugo.toml` — site config, menus, permalinks
- `content/blog/` — blog posts (`title`, `date`, `tags` front matter)
- `content/*.md` — standalone pages (Experiments, Bookmarks, My Setup, …)
- `static/assets/` — images
- `static/**/*.html` — legacy redirect stubs for pre-Hugo `.html` URLs

Post URLs are `/posts/<slug>/`; tag pages live under `/tags/<tag>/`.

## Deploy

Pushes to `main` build with Hugo via
`.github/workflows/deploy.yml` and publish `public/` to GitHub Pages.
