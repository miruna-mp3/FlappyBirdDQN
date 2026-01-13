# Flappy Bird DQN

Un agent care învață să joace Flappy Bird folosind Deep Q-Learning pe bază de pixeli.

## Ce face proiectul

Agentul primește ca input imaginile din joc (pixeli), le procesează și învață singur cum să joace.

**Rezultat final**: După ~2400 episoade de antrenare (~46 minute pe GPU), agentul atinge un reward mediu de **123.9**, cu episoade individuale de până la **263 puncte**.

## Mediul de joc

Flappy Bird este un joc simplu dar dificil:

| Aspect | Detalii |
|--------|---------|
| Obiectiv | Treci prin spațiile dintre tuburi fără să te lovești |
| Acțiuni | 2 posibile: nu face nimic (cade) sau sari |
| Input original | Imagine RGB 512×288 pixeli |
| Terminare | Lovire de tub, sol sau tavan |

Rewards din mediu:
- +0.1 pentru fiecare frame în care supraviețuiești
- +1.0 pentru fiecare tub trecut
- -1.0 la moarte

## Preprocesarea imaginilor

Imaginile RGB originale sunt prea mari și conțin prea multă informație inutilă. Le transformăm astfel:

| Pas | Transformare | De ce |
|-----|--------------|-------|
| 1 | RGB → Grayscale | Culoarea nu ajută - pasărea și tuburile sunt vizibile și fără culoare |
| 2 | 512×288 → 84×84 | Reducem drastic numărul de pixeli, dar păstrăm suficiente detalii |
| 3 | Binarizare (prag 150) | Eliminăm fundalul complet - rămân doar pasărea și tuburile ca siluete albe pe fundal negru |
| 4 | Stivă de 4 frame-uri | Un singur frame nu arată direcția mișcării - cu 4 frame-uri agentul "vede" viteza și traiectoria |

**Rezultat**: input de forma **(4, 84, 84)** - 4 imagini binare de 84×84 pixeli

De ce binarizare? Fundalul din Flappy Bird (cerul, norii, solul) este doar zgomot vizual. Prin binarizare, rețeaua vede doar ce contează: poziția pasării și a tuburilor.

## Rețeaua neuronală

Folosim o rețea convoluțională (CNN) pentru că aceasta poate extrage automat features din imagini (margini, forme, poziții).

### Arhitectura

```
Input: (batch, 4, 84, 84)
         ↓
    Conv 8×8, stride 4 → 32 canale + ReLU
         ↓
    Conv 4×4, stride 2 → 64 canale + ReLU
         ↓
    Conv 3×3, stride 1 → 64 canale + ReLU
         ↓
    Flatten → 3136 neuroni
         ↓
    Fully Connected → 512 neuroni + ReLU
         ↓
Output: 2 valori (Q-value pentru "stai" și "sari")
```

De ce această structură?
- **3 straturi convoluționale**: suficiente pentru un joc simplu vizual ca Flappy Bird
- **Kernel-uri descrescătoare (8→4→3)**: primul strat vede patterns mari (tuburi întregi), ultimul vede detalii fine (margini precise)
- **512 neuroni în FC**: un compromis bun între capacitate și simplitate

## Algoritmul DQN

Deep Q-Network combină Q-learning clasic cu rețele neuronale.

### Ideea de bază

Agentul învață o funcție Q(stare, acțiune) care estimează "cât de bună" este o acțiune într-o anumită stare. La fiecare pas, alege acțiunea cu Q-value maxim.

Ecuația fundamentală (Bellman):
```
Q(s, a) = reward + γ × max Q(s', a')
```
Unde γ (gamma) = 0.99 reprezintă cât de mult contează recompensele viitoare.

### Componente cheie

**1. Experience Replay Buffer**

În loc să învețe imediat din fiecare experiență, agentul stochează tranzițiile (stare, acțiune, reward, stare_următoare) într-un buffer și învață din sample-uri aleatorii.

Experiențele consecutive sunt foarte corelate (10 frame-uri la rând arată aproape la fel). Sampling aleator din buffer elimină această corelație și stabilizează antrenarea.

| Parametru | Valoare | Explicație |
|-----------|---------|------------|
| Capacitate | 50,000 | Suficient pentru diversitate, nu prea mare pentru memorie |
| Batch size | 32 | standard, mai mare ar fi ineficient |

**2. Target Network**

Avem două rețele: una care învață (policy network) și una "înghețată" (target network) folosită pentru calculul target-ului.

Fără target network, target-ul se schimbă la fiecare pas de antrenare, ceea ce destabilizează învățarea (e ca și cum ai încerca să nimerești o țintă care se mișcă constant).

| Parametru | Valoare | Explicație |
|-----------|---------|------------|
| Update frequency | 1000 pași | Target network se actualizează rar |
| Tau (τ) | 0.001 | Soft update: target = τ×policy + (1-τ)×target |

**3. Explorare vs Exploatare**

La început, agentul explorează aleatoriu. Treptat, începe să folosească ce a învățat.

| Parametru | Valoare | Explicație |
|-----------|---------|------------|
| Epsilon start | 0.1 | Începem cu 10% explorare |
| Epsilon end | 0.001 | La final, aproape 0% explorare |
| Decay | 200,000 pași | Scădere liniară |

Flappy Bird penalizează dur acțiunile random. Cu prea multă explorare, agentul moare instant și nu învață nimic util. Am observat că 10% explorare inițială funcționează mult mai bine.

**4. Explorare cu bias**

Când explorează aleatoriu, agentul alege:
- 85% să stea (nu sare)
- 15% să sară

De ce? În Flappy Bird, a sări prea des te omoară rapid. Majoritatea timpului trebuie să aștepți momentul potrivit. Acest bias ajută agentul să supraviețuiască mai mult în timpul explorării și să colecteze experiențe mai utile.

**5. Faza de observare**

Primii 10,000 de pași, agentul doar colectează experiențe fără să învețe.

Replay buffer-ul trebuie să aibă suficiente experiențe diverse înainte de a începe antrenarea. Dacă începem prea devreme, agentul învață dintr-un set mic și nereprezentativ de date.

## Hiperparametri

| Parametru | Valoare | Alegere și motivație |
|-----------|---------|---------------------|
| Learning rate | 3e-5 | Mic pentru stabilitate - DQN pe imagini e sensibil |
| Gamma (γ) | 0.99 | Standard - recompensele viitoare contează mult |
| Batch size | 32 | Clasic pentru DQN |
| Buffer capacity | 50,000 | Echilibru memorie/diversitate |
| Target update | 1000 pași | Mai rar = mai stabil |
| Tau | 0.001 | Soft update gradual |
| Observation steps | 10,000 | ~400 episoade de date înainte de antrenare |
| Frame skip | 2 | Acțiunea se repetă 2 frame-uri |
| Loss function | Huber (Smooth L1) | Mai robust la outliers decât MSE |
| Optimizer | Adam | Standard, convergență rapidă |

## Rezultatele antrenării

Antrenare pe 2410 episoade, GPU CUDA, timp total ~46 minute.

### Evoluția performanței

```
Avg Reward
    │
140 ┤                                                            ▄█
    │                                                           ▄██
120 ┤                                                          ▄███
    │                                                         ▄████
100 ┤                                                        ▄█████
    │                                                       ▄██████
 80 ┤                                                      ▄███████
    │                                                     ▄████████
 60 ┤                                                   ▄██████████
    │                                                  ▄███████████
 40 ┤                                          ▄▄▄▄▄▄▄████████████
    │                                    ▄▄▄▄▄█████████████████████
 20 ┤                           ▄▄▄▄▄▄▄▄████████████████████████████
    │                  ▄▄▄▄▄▄▄██████████████████████████████████████
  0 ┼▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄███████████████████████████████████████████████
    └────────────────────────────────────────────────────────────────
     0    400    800   1200   1600   2000   2400              Episod
          │             │             
         OBS          TRN        
```

### Momente cheie

| Episod | Fază | Avg Reward | Observații |
|--------|------|------------|------------|
| 1-400 | OBS | ~2.3 | Faza de observare - doar colectare date |
| 410 | TRN | 2.3 | Începe antrenarea propriu-zisă |
| 700 | TRN | 3.5 | Primele semne de învățare |
| 1000 | TRN | 6.5 | Progres constant |
| 1500 | TRN | 13.9 | Episod individual de 49.7 puncte |
| 2000 | TRN | 23.5 | Se apropie de obiectiv |
| 2130 | TRN | **30.3** | **Obiectivul de 30 puncte atins** |
| 2200 | TRN | 37.5 | Depășește clar obiectivul |
| 2370 | TRN | 67.2 | Cel mai bun episod: 263 puncte, 1497 pași |
| 2410 | TRN | **123.9** | Performanță finală |

### Extras din log

```
  [ FLAPPY BIRD DQN ]

  Dispozitiv       cuda
  Episoade         3000
  Fază observare   10000 pași

  [ 400] [OBS]  [Reward]    2.0  [Avg]    2.3  [Buffer]  9950
  [ 410] [TRN]  [Reward]   -1.6  [Avg]    2.3  [ε] 0.0950   ← începe antrenarea
  ...
  [1000] [TRN]  [Reward]    6.3  [Avg]    6.5  [ε] 0.0862
  ...
  [2130] [TRN]  [Reward]   27.7  [Avg]   30.3  [ε] 0.0365   ← obiectiv atins
          [ NOU RECORD 30.3 ]
  ...
  [2370] [TRN]  [Reward]  263.0  [Avg]   67.2  [ε] 0.0020   ← cel mai bun episod
          [ NOU RECORD 67.2 ]
  ...
  [2410] [TRN]  [Reward]   53.0  [Avg]  123.9  [ε] 0.0010   ← final
          [ NOU RECORD 123.9 ]
```

## Structura fișierelor

| Fișier | Rol |
|--------|-----|
| `train_v2.py` | Script principal de antrenare |
| `dqn_agent.py` | Implementarea agentului DQN |
| `model.py` | Arhitectura CNN |
| `replay_buffer.py` | Experience replay buffer |
| `env_wrapper.py` | Preprocesarea imaginilor |
| `play.py` | Evaluare și vizualizare agent antrenat |

## Utilizare

**Antrenare:**
```bash
python train_v2.py
```

**Evaluare model antrenat:**
```bash
python play.py --model best_flappy_dqn.pth --episodes 10
```

## Concluzii

Agentul a învățat cu succes să joace Flappy Bird doar din pixeli, fără nicio cunoaștere prealabilă a regulilor jocului.

Factori cheie pentru succes:
- **Binarizarea input-ului**: eliminarea fundalului a simplificat semnificativ problema
- **Explorare cu bias**: agentul supraviețuiește mai mult și colectează experiențe utile
- **Epsilon mic inițial**: explorare excesivă ucide agentul instant în Flappy Bird
- **Faza de observare**: buffer-ul trebuie populat înainte de antrenare

Obiectivul de 30 puncte a fost atins la episodul 2130, iar performanța finală de 123.9 depășește cu mult acest prag.
