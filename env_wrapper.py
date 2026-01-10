import gymnasium as gym
import numpy as np
from collections import deque
from PIL import Image


class FlappyBirdWrapper(gym.Wrapper):
    """
    Wrapper pentru preprocesarea input-ului Flappy Bird:
    - Conversie la grayscale
    - Resize la 84x84
    - Stack de 4 frame-uri
    - Normalizare la [0, 1]
    """
    
    def __init__(self, env, img_size=84, stack_frames=4):
        super().__init__(env)
        self.img_size = img_size
        self.stack_frames = stack_frames
        
        # Deque pentru stocarea frame-urilor
        self.frames = deque(maxlen=stack_frames)
        
        # Actualizează observation space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(stack_frames, img_size, img_size),
            dtype=np.float32
        )
    
    def _preprocess_frame(self, frame):
        """Conversie RGB -> Grayscale -> Resize -> Normalize"""
        # Conversie la grayscale
        img = Image.fromarray(frame).convert('L')
        
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Conversie la numpy și normalizare
        frame = np.array(img, dtype=np.float32) / 255.0
        
        return frame
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Preprocesează frame-ul inițial
        frame = self._preprocess_frame(obs)
        
        # Umple stack-ul cu același frame
        for _ in range(self.stack_frames):
            self.frames.append(frame)
        
        return self._get_stacked_frames(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Preprocesează și adaugă noul frame
        frame = self._preprocess_frame(obs)
        self.frames.append(frame)
        
        # Reward shaping: încurajează supraviețuirea
        # Reward original: +0.1 per frame, +1.0 per tub trecut, -1000 la moarte
        # Facem reward-ul mai pozitiv pentru învățare
        # Reward shaping mai agresiv
        shaped_reward = reward
        if terminated:
            shaped_reward = -1  # Penalizare mica
        elif reward > 1.0:
            shaped_reward = 10  # Bonus mare pentru tub trecut!
        else:
            shaped_reward = 0.1  # Supraviețuire
        
        return self._get_stacked_frames(), shaped_reward, terminated, truncated, info
    
    def _get_stacked_frames(self):
        """Returnează stack-ul de frame-uri ca array numpy"""
        return np.stack(self.frames, axis=0)