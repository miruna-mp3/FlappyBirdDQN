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