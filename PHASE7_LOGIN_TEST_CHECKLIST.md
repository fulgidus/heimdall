# 🔐 PHASE 7: Login Testing Guide

**Server**: http://localhost:5173  
**Credenziali Default**: 
- Email: `admin@heimdall.local`
- Password: `Admin123!@#`

---

## ✅ Checklist di Verifica

### 1. Layout Desktop
- [ ] La login page è **centrata** (non occupa tutta la pagina)
- [ ] Il card ha un **bordo definito di 2px** in Sea Green
- [ ] Il background è un **gradient solido** da Oxford Blue
- [ ] Max-width è rispettato (~448px)

### 2. Accessibilità (WCAG AA)
- [ ] Il testo è **facilmente leggibile** (contrasto alto)
- [ ] I label sono in **Light Green** (emerald - AAA)
- [ ] I placeholder sono in **Sea Green** (AA)
- [ ] I bordi sono in **Sea Green** (AA, definiti)
- [ ] Il pulsante "Sign In" è in **Light Green** su Oxford Blue (AAA)
- [ ] Zoom a 200%: il testo rimane **leggibile senza scrollbar orizzontale**
- [ ] Color blind test: i colori sono **distinguibili senza solo il colore**

### 3. Funzionalità Login - Caso Valido
- [ ] Inserisci: `admin@heimdall.local`
- [ ] Inserisci password: `Admin123!@#`
- [ ] Clicca "Sign In"
- [ ] Vieni **reindirizzato a /dashboard** (no errori)
- [ ] Il token JWT è salvato in localStorage (DevTools → Storage → Local Storage)

### 4. Funzionalità Login - Caso Errore
- [ ] Inserisci: `admin@heimdall.local`
- [ ] Inserisci password: `wrongpassword`
- [ ] Clicca "Sign In"
- [ ] Appare **error message in rosso**: "⚠️ Invalid email or password"
- [ ] **Non** vieni reindirizzato
- [ ] Puoi correggere e riprovare

### 5. Funzionalità Login - Email Valida, Password Sbagliata
- [ ] Inserisci: `admin@heimdall.local`
- [ ] Inserisci password: `12345678`
- [ ] Clicca "Sign In"
- [ ] Appare errore
- [ ] Message è **chiaramente visibile** (contrasto AA)

### 6. Funzionalità Login - Email Sbagliata, Password Valida
- [ ] Inserisci: `wrong@example.com`
- [ ] Inserisci password: `Admin123!@#`
- [ ] Clicca "Sign In"
- [ ] Appare errore (email non corrisponde)

### 7. UX - Show/Hide Password
- [ ] Digita la password
- [ ] Clicca sull'icona **occhio** (Eye icon)
- [ ] La password diventa **visibile** (cambia da `••••` a testo)
- [ ] Clicca di nuovo
- [ ] La password torna **nascosta**

### 8. UX - Remember Me
- [ ] Seleziona "Remember me" checkbox
- [ ] Accedi con credenziali valide
- [ ] Refresh la pagina
- [ ] Sei **ancora loggato** (localStorage persiste)

### 9. UX - Demo Credentials Box
- [ ] Scorri in basso dopo il form
- [ ] Appare un box con sfondo Sea Green
- [ ] Mostra: 
  ```
  📋 Demo Credentials
  Email: admin@heimdall.local
  Password: Admin123!@#
  ```
- [ ] Il testo è **leggibile** (contrasto AA)

### 10. Responsive - Mobile (375px)
- [ ] Apri DevTools (F12)
- [ ] Seleziona "iPhone 12/13" (375px width)
- [ ] La login page è **centrata** e **leggibile**
- [ ] Il card si **adatta** senza overflow
- [ ] Buttons e input hanno **dimensioni toccabili** (44px+ height)
- [ ] Nessun scroll orizzontale

### 11. Responsive - Tablet (768px)
- [ ] Seleziona "iPad" (768px width)
- [ ] La login page rimane **centrata**
- [ ] Max-width è rispettato (448px)

### 12. Mobile - Show Password su Touch
- [ ] Su mobile, tap l'icona eye
- [ ] La password viene **mostrata/nascosta** senza lag
- [ ] Non appare keyboard virtuale (button, non input)

### 13. Accessibility - Keyboard Navigation
- [ ] Premi **Tab** dalla barra dell'indirizzo
- [ ] Focus entra su **Email input** (visualizza border)
- [ ] Premi **Tab** → Focus su **Password input**
- [ ] Premi **Tab** → Focus su **Remember me checkbox**
- [ ] Premi **Tab** → Focus su **Forgot password link**
- [ ] Premi **Tab** → Focus su **Sign In button**
- [ ] Premi **Enter** → Submits il form

### 14. Accessibility - Screen Reader (NVDA/JAWS)
- [ ] Screen reader annuncia: "Welcome Back heading"
- [ ] Annuncia: "Email Address label input"
- [ ] Annuncia: "Password label input"
- [ ] Annuncia: "Remember me checkbox"
- [ ] Annuncia: "Sign In button"

### 15. Loading State
- [ ] Accedi con credenziali valide
- [ ] Prima che redirect avvenga, il button mostra: "⏳ Signing In..."
- [ ] Button è **disabilitato** (disabled attribute)
- [ ] Puoi **osservare lo spinner** animato

### 16. Error Styling
- [ ] Inserisci credenziali errate
- [ ] L'error message appare con:
  - [ ] **⚠️ emoji**
  - [ ] Background **rosso solido** (bg-red-600)
  - [ ] Bordo **rosso** (border-2 border-red-400)
  - [ ] Testo **bianco** (text-white)
  - [ ] Font **bold** (font-semibold)

### 17. Color Contrasts (Chrome DevTools)
- [ ] Apri DevTools (F12)
- [ ] Vai a "Elements" tab
- [ ] Inspect il button "Sign In"
- [ ] Nella sezione "Styles", Chrome mostra il contrast ratio
- [ ] Dovrebbe mostrare **AAA** (>7:1)

---

## 🧪 Comandi Utili

### Visualizzare le credenziali nel .env
```bash
cat c:\Users\aless\Documents\Projects\heimdall\frontend\.env
```

### Modificare le credenziali
```bash
# Windows PowerShell
@"
VITE_ADMIN_EMAIL=newadmin@example.com
VITE_ADMIN_PASSWORD=NewPassword123!
VITE_API_URL=http://localhost:8000/api
VITE_API_TIMEOUT=10000
VITE_ENV=development
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_DEBUG=false
VITE_AUTH_TOKEN_KEY=heimdall_auth_token
VITE_SESSION_TIMEOUT=3600000
VITE_SOCKET_URL=ws://localhost:8000/ws
"@ | Set-Content c:\Users\aless\Documents\Projects\heimdall\frontend\.env
```

### Verifica nel localStorage (Console)
```javascript
// Browser Console
console.log(localStorage.getItem('auth-store'));
```

### Test della login API
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@heimdall.local","password":"Admin123!@#"}'
```

---

## 🔴 Problemi Comuni e Soluzioni

### "Invalid email or password" appare sempre
**Causa**: Le credenziali nel .env non corrispondono a quello che inserisci
**Soluzione**: 
```bash
cat .env | grep VITE_ADMIN
# Copia esattamente email e password da qui
```

### Il password input non mostra/nasconde la password
**Causa**: showPassword state non aggiornato
**Soluzione**: Refresh il browser (F5)

### La login page occupa tutta la pagina
**Causa**: Stile non aggiornato (cache)
**Soluzione**: 
```bash
# Hard refresh
Ctrl+Shift+R (Windows) o Cmd+Shift+R (Mac)
```

### Il contrasto è ancora basso
**Causa**: Browser usando stili cachati
**Soluzione**:
```bash
npm run dev  # Riparte il dev server
# Hard refresh nel browser
```

### Error message non visibile
**Causa**: Colore di sfondo troppo scuro
**Soluzione**: Verificare che il background sia `bg-red-600` (non opacity)
```tsx
<div className="p-4 bg-red-600 border-2 border-red-400 rounded-lg text-white font-semibold">
```

---

## 📸 Screenshot Expected Behavior

### Desktop View (1920x1080)
```
┌─────────────────────────────────────────┐
│                                         │
│                  🚀 Heimdall           │
│              RF Localization Platform    │
│                                         │
│    ┌─────────────────────────────────┐  │
│    │    Welcome Back                 │  │
│    │  Sign in to your account        │  │
│    │                                 │  │
│    │  Email Address                  │  │
│    │  [admin@heimdall.local]         │  │
│    │                                 │  │
│    │  Password                       │  │
│    │  [••••••••] [👁️]               │  │
│    │                                 │  │
│    │  ☑️ Remember me  Forgot?       │  │
│    │                                 │  │
│    │    [ Sign In ]                  │  │
│    │                                 │  │
│    │    Google  GitHub  Microsoft    │  │
│    │                                 │  │
│    │  Don't have account? Sign up    │  │
│    └─────────────────────────────────┘  │
│                                         │
│  ┌─────────────────────────────────────┐ │
│  │ 📋 Demo Credentials                 │ │
│  │ Email: admin@heimdall.local         │ │
│  │ Password: Admin123!@#               │ │
│  └─────────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

### Mobile View (375x812)
```
┌──────────────────┐
│                  │
│ 🚀 Heimdall     │
│ RF Localization  │
│                  │
│ ┌──────────────┐ │
│ │ Welcome Back │ │
│ │ Sign in      │ │
│ │              │ │
│ │ Email        │ │
│ │ [input]      │ │
│ │              │ │
│ │ Password     │ │
│ │ [••••] [👁️]  │ │
│ │              │ │
│ │ ☑️ Remember  │ │
│ │              │ │
│ │ [ Sign In ]  │ │
│ │              │ │
│ │ 🔍 🐙 ⊞     │ │
│ │              │ │
│ │ Sign up link │ │
│ └──────────────┘ │
│                  │
│ ┌──────────────┐ │
│ │ 📋 Demo      │ │
│ │ Email: ...   │ │
│ │ Pass: ...    │ │
│ └──────────────┘ │
│                  │
└──────────────────┘
```

---

## ✅ Sign Off

Una volta che tutti i test in questa checklist passano, la login page è **PRONTA PER PRODUZIONE** ✅

Segnala i risultati:
```
✅ All 17 test categories passed
✅ WCAG AA compliance verified
✅ Responsive design confirmed
✅ Ready for Phase 8: Backend Integration
```
