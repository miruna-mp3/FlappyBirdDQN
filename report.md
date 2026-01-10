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