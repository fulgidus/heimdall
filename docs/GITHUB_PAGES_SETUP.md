# 🚀 Abilitazione GitHub Pages per Coverage Reports

## Problema
Il workflow di coverage pubblica su GitHub Pages, ma il branch `gh-pages` non è ancora configurato come sorgente.

## Soluzione: Abilitare GitHub Pages

### Step 1: Vai alle Impostazioni del Repository

1. Apri: https://github.com/fulgidus/heimdall/settings/pages
2. Oppure: **Settings** → **Pages** (sidebar sinistra)

### Step 2: Configura la Sorgente

Nella sezione **"Build and deployment"**:

1. **Branch source:** seleziona `gh-pages`
2. **Folder:** seleziona `/` (root)
3. Clicca **Save**

**Output atteso:**
```
✅ Your site is live at https://fulgidus.github.io/heimdall
```

### Step 3: Struttura risultante

Dopo l'abilitazione, i tuoi report saranno disponibili a:

```
https://fulgidus.github.io/heimdall/
├── coverage/
│   ├── develop/          ← Ultimo report da develop
│   │   ├── index.html
│   │   ├── badge.svg
│   │   ├── backend/
│   │   └── frontend/
│   └── main/             ← Ultimo report da main
│       ├── index.html
│       ├── badge.svg
│       ├── backend/
│       └── frontend/
└── (altri file di Pages)
```

### URL Finali

- **Coverage Develop:** https://fulgidus.github.io/heimdall/coverage/develop
- **Coverage Main:** https://fulgidus.github.io/heimdall/coverage/main
- **Badge Develop:** https://raw.githubusercontent.com/fulgidus/heimdall/gh-pages/coverage/develop/badge.svg

### Step 4: Verifica nel Workflow

Dopo l'abilitazione, il prossimo push a `develop` attiveràil workflow:

1. ✅ Esegue test (backend + frontend)
2. ✅ Genera report HTML + badge SVG
3. ✅ Pubblica a `gh-pages/coverage/develop`
4. ✅ Disponibile a https://fulgidus.github.io/heimdall/coverage/develop

## Troubleshooting

### Errore: "gh-pages branch doesn't exist"
- **Soluzione:** Il workflow crea il branch automaticamente al primo push. Se non esiste ancora:
  ```bash
  git push origin --allow-empty-message -m "" develop
  ```
  Questo triggerà il workflow, che creerà `gh-pages`.

### Errore: "Failed to deploy"
- **Check:** Vai a https://github.com/fulgidus/heimdall/settings/pages e verifica:
  1. Branch: `gh-pages` ✓
  2. Folder: `/` ✓
  3. Salva se necessario

### Reports non visibili
- **Attendi:** I report sono public dopo ~5 minuti dal push
- **URL:** Usa sempre il formato: `https://fulgidus.github.io/heimdall/coverage/develop`

## Comandi Utili

```bash
# Verifica se gh-pages esiste
git branch -r | grep gh-pages

# Vedi l'ultimo deploy (se attivato)
git log --oneline gh-pages | head -5

# Visualizza il contenuto di gh-pages
git show gh-pages:coverage/develop/index.html | head -20
```

## Badge nel README

Una volta abilitato Pages, il badge nel README si aggiornerà automaticamente:

```markdown
[![Coverage](https://raw.githubusercontent.com/fulgidus/heimdall/gh-pages/coverage/develop/badge.svg)](https://fulgidus.github.io/heimdall/coverage/develop)
```

---

**Dopo l'abilitazione:** Il workflow funzionerà perfettamente e i tuoi report saranno visibili a chiunque! 🎉
