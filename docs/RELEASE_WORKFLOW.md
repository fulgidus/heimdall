# Release Workflow

## Panoramica

Il progetto Heimdall utilizza un workflow GitHub Actions completamente automatizzato per gestire le release. Il processo è semplice e richiede solo il tagging delle versioni su `develop`.

## Come fare una release

### Prerequisiti
- Essere su `develop`
- I vostri cambiamenti sono già committati e pushati

### Procedura

#### 1. Aggiornare il CHANGELOG
Modificare il `CHANGELOG.md` e spostare i vostri cambiamenti dalla sezione `[Unreleased]` a una nuova sezione `[VERSION]`:

```markdown
## [Unreleased]

### Added
- Nothing yet

### Fixed
- Nothing yet

---

## [0.3.0] - 2025-10-30

### Added
- **Feature Name**
  - Descrizione della feature
  - Dettagli implementativi
```

#### 2. Aggiornare le versioni (opzionale, la workflow lo fa automaticamente dopo)

In locale, aggiornare:
- `pyproject.toml`: `version = "0.3.0"`
- `frontend/package.json`: `"version": "0.3.0"`
- `CHANGELOG.md`: Creare sezione `[0.3.0]` con data odierna

#### 3. Committare e taggare

```bash
# Committare i cambiamenti
git add pyproject.toml frontend/package.json CHANGELOG.md
git commit -m "Release v0.3.0 - Feature description"

# Creare il tag (FORMATO IMPORTANTE: vX.Y.Z)
git tag -a v0.3.0 -m "Release 0.3.0 - Feature description"

# Pushare a origin
git push origin develop v0.3.0
```

### Cosa accade automaticamente

Una volta che il tag viene pushato:

1. **Create Release** (Job 1)
   - ✅ Estrae il numero di versione dal tag
   - ✅ Legge il CHANGELOG.md
   - ✅ Crea una GitHub Release con:
     - Tag name: `v0.3.0`
     - Title: `Release v0.3.0`
     - Body: Contenuto della sezione `[0.3.0]` del CHANGELOG

2. **Merge to Main** (Job 2 - dipende da Job 1)
   - ✅ Checkoutare `main`
   - ✅ Merge di `develop` in `main`
   - ✅ Push di `main`

3. **Update Version** (Job 3 - dipende da Job 2)
   - ✅ Incrementa la versione patch (0.3.0 → 0.3.1)
   - ✅ Aggiorna `pyproject.toml`
   - ✅ Aggiorna `frontend/package.json`
   - ✅ Rigenera `CHANGELOG.md` con nuova sezione `[Unreleased]`
   - ✅ Committa e pushes i cambiamenti in `develop`

## Struttura del Tag

I tag devono seguire il formato **Semantic Versioning**:
- `v0.3.0` ✅ (CORRETTO)
- `v1.2.3` ✅ (CORRETTO)
- `0.3.0` ❌ (SBAGLIATO - manca la 'v')
- `release-0.3.0` ❌ (SBAGLIATO - formato non riconosciuto)

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ git tag -a v0.3.0                                           │
│ git push origin develop v0.3.0                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   GitHub Actions Triggered     │
        └────────────┬───────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
    ┌─────────┐            ┌──────────┐
    │ Job 1   │            │ Job 1    │
    │ Create  │─────>wait  │ Push to  │
    │Release  │            │ GitHub   │
    └─────────┘            └──────────┘
         │
         ▼ (needs: create-release)
    ┌─────────────┐
    │ Job 2       │
    │ Merge to    │
    │ main        │
    └──────┬──────┘
           │
           ▼ (needs: merge-to-main)
    ┌──────────────────┐
    │ Job 3            │
    │ Update Version   │
    │ in develop       │
    └──────────────────┘
           │
           ▼
    ✅ Release Complete!
    - GitHub Release created
    - main synced with develop
    - Next development version ready
```

## Verificare la Release

### Su GitHub
1. Vai a [Releases](https://github.com/fulgidus/heimdall/releases)
2. La release `v0.3.0` dovrebbe essere visibile con il changelog

### Su Local
```bash
# Verificare che main sia sincronizzato
git checkout main
git log --oneline -3

# Dovrebbe mostrare il merge commit
# Commit Release v0.3.0

# Verificare il tag su main
git tag --points-at HEAD
```

## Troubleshooting

### Il tag non triggerava la workflow

**Possibili cause:**
- Il tag non segue il formato `v*`
- Il tag è stato creato senza il flag `-a` (non è annotato)
- Il branch non è `develop`

**Soluzione:**
```bash
# Eliminare il tag locale
git tag -d v0.3.0

# Eliminare dal remote
git push origin :v0.3.0

# Ricreare il tag correttamente
git tag -a v0.3.0 -m "Release 0.3.0"
git push origin v0.3.0
```

### La workflow fallisce

**Verificare i logs:**
1. Vai a GitHub Actions
2. Clicca sulla release workflow
3. Controlla i job che hanno fallito

**Errori comuni:**
- **sed command failed**: Il file non ha il formato atteso
  - Soluzione: Verificare che `pyproject.toml` e `package.json` abbiano la sintassi corretta
  
- **Git merge conflict**: develop e main hanno diverged
  - Soluzione: Risolvere il conflitto manualmente su main

- **Token permissions**: La workflow non ha i permessi
  - Soluzione: Verificare che `GITHUB_TOKEN` abbia i permessi di `write` su `contents` e `pull-requests`

## Best Practices

1. **Sempre aggiornare il CHANGELOG prima del release**
   - Renderà il rilascio più efficace
   - Il changelog sarà automaticamente nel release su GitHub

2. **Usare message significative nel tag**
   ```bash
   git tag -a v0.3.0 -m "Release 0.3.0 - Phase 7 complete with automated release workflow"
   ```

3. **Non pushare direttamente a main**
   - La workflow si occupa del merge
   - Mantiene l'integrità del branch

4. **Testare in develop prima di taggare**
   - Verificare che tutti i test passino
   - Fare un review dei cambiamenti principali

## Future Enhancements

- [ ] Creare automaticamente GitHub Releases con i file binari
- [ ] Notificare Slack/Discord al rilascio
- [ ] Automaticamente uploadare i Docker images
- [ ] Generare release notes con PRs associate
- [ ] Creare changelog entry automaticamente dai commit messages
