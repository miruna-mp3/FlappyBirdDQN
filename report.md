# Flappy Bird DQN - Raport Tehnic

## 1. Introducere

### 1.1 Obiectiv
Implementarea unui agent de reinforcement learning pentru jocul Flappy Bird folosind algoritmul DQN (Deep Q-Network) cu input pe pixeli. Scopul este antrenarea unui agent capabil să obțină un punctaj de peste 30 de puncte.

### 1.2 Mediul de lucru
- **Joc**: Flappy Bird
- **Framework mediu**: flappy-bird-gymnasium (compatibil Gymnasium/OpenAI Gym)
- **Framework deep learning**: PyTorch
- **Limbaj**: Python 3.x

### 1.3 Descrierea mediului

Mediul Flappy Bird este un joc 2D în care:
- **Obiectiv**: Pasărea trebuie să treacă prin tuburi fără să se lovească
- **Spațiu de acțiuni**: 2 acțiuni discrete
  - `0`: Nu face nimic (pasărea cade)
  - `1`: Jump (pasărea sare)
- **Observații**: Imagini RGB de dimensiune variabilă (tipic 512×288×3)
- **Reward**: 
  - +0.1 pentru fiecare frame supraviețuit
  - +1.0 pentru fiecare tub trecut
- **Episod terminat**: Când pasărea se lovește de tub sau de sol/tavan

### 1.4 Setup verificat
```bash
pip install -r requirements.txt
python test_env.py
```

Mediul rulează corect și oferă observații de tip imagine RGB.

## 2. Preprocesarea Input-ului

### 2.1 Motivație
Imaginile RGB originale (512×288×3) sunt prea mari și conțin informații redundante pentru învățare. Preprocesarea reduce dimensiunea input-ului și păstrează doar informația esențială.

### 2.2 Pipeline de preprocesare

Fiecare frame trece prin următoarele transformări:

1. **Conversie la Grayscale**
   - RGB (3 canale) → Grayscale (1 canal)
   - Culoarea nu este esențială pentru joc

2. **Resize la 84×84 pixeli**
   - Reduce dimensiunea de la 512×288 la 84×84
   - Păstrează proporțiile relevante ale jocului
   - Reduce numărul de parametri ai rețelei

3. **Normalizare**
   - Pixeli de la [0, 255] → [0, 1]
   - Facilitează antrenarea rețelei neuronale

4. **Frame Stacking**
   - Se stochează ultimele 4 frame-uri consecutive
   - Oferă informație despre mișcare și viteză
   - Input final: **(4, 84, 84)**

### 2.3 Implementare

Preprocesarea este implementată printr-un wrapper Gymnasium (`FlappyBirdWrapper`) care:
- Aplică automat transformările la fiecare `reset()` și `step()`
- Menține un buffer (deque) cu ultimele 4 frame-uri
- Returnează stack-ul ca input pentru agent

### 2.4 Verificare

```bash
python test_wrapper.py
```

Output final: **shape (4, 84, 84), dtype float32, valori în [0, 1]**

## 3. Arhitectura Rețelei Neuronale

### 3.1 Tip de rețea
Convolutional Neural Network (CNN) - optimă pentru procesarea imaginilor.

### 3.2 Structura modelului

**Input:** `(batch, 4, 84, 84)` - 4 frame-uri grayscale de 84×84 pixeli

**Convolutional Layers:**
1. Conv1: `4 → 32` channels, kernel 8×8, stride 4, + ReLU
2. Conv2: `32 → 64` channels, kernel 4×4, stride 2, + ReLU
3. Conv3: `64 → 64` channels, kernel 3×3, stride 1, + ReLU

**Fully Connected Layers:**
1. FC1: `conv_output → 512` neuroni, + ReLU
2. FC2: `512 → 2` neuroni (Q-values pentru cele 2 acțiuni)

**Output:** `(batch, 2)` - Q-values pentru acțiunile `[do_nothing, jump]`

### 3.3 Alegeri de design

- **3 conv layers**: Suficient pentru extragerea de features din Flappy Bird (joc simplu vizual)
- **Kernel sizes descrescătoare**: (8→4→3) pentru captarea de patterns la diferite scale
- **ReLU activation**: Standard pentru CNN-uri, convergență rapidă
- **512 neuroni în FC1**: Balans între capacitate și simplitate
- **Fără dropout/batch norm**: DQN funcționează bine fără acestea pentru jocuri simple

### 3.4 Parametri

Modelul conține aproximativ **2-3 milioane** de parametri antrenabili, suficienți pentru task-ul dat fără risc major de overfitting (datorită replay buffer-ului).

### 3.5 Verificare

```bash
python test_model.py
```

Modelul acceptă input de shape `(batch, 4, 84, 84)` și produce output `(batch, 2)`.

## 4. Experience Replay Buffer

### 4.1 Motivație

În Q-learning clasic, agentul învață imediat din fiecare experiență (s, a, r, s'). Acest lucru creează două probleme:
- **Corelație temporală**: Experiențele consecutive sunt foarte similare
- **Ineficiență**: Fiecare experiență este folosită o singură dată

Experience Replay rezolvă ambele probleme.

### 4.2 Principiu de funcționare

Replay Buffer-ul este o structură de tip **FIFO (First-In-First-Out)** cu capacitate fixă:
1. Stochează tranziții `(state, action, reward, next_state, done)`
2. Când e plin, cele mai vechi tranziții sunt șterse automat
3. La fiecare pas de antrenare, se sample aleator un **batch** de tranziții
4. Rețeaua învață din acest batch, nu din experiențe consecutive

### 4.3 Avantaje

- **Decorelează experiențele**: Sampling aleator elimină corelația temporală
- **Reutilizare**: Fiecare experiență poate fi folosită în multiple batch-uri
- **Stabilitate**: Gradient-ul este mai stabil (mediere peste experiențe diverse)
- **Eficiență**: Învață mai repede din același număr de interacțiuni cu mediul

### 4.4 Implementare

Clasa `ReplayBuffer`:
- **Capacitate**: 100,000 tranziții (configurabil)
- **Structură**: `collections.deque` (efficient pentru FIFO)
- **Metode**:
  - `push(s, a, r, s', done)`: Adaugă tranziție
  - `sample(batch_size)`: Returnează batch aleator
  - `__len__()`: Număr curent de tranziții

### 4.5 Parametri

- **Capacitate buffer**: 100,000 tranziții
- **Batch size**: 32 (standard pentru DQN)
- **Tip stocare**: numpy arrays (float32 pentru state/reward, int64 pentru action)

### 4.6 Verificare

```bash
python test_replay_buffer.py
```

Buffer-ul stochează și samplează corect tranziții, respectând capacitatea maximă.