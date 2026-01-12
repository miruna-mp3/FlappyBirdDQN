import gymnasium as gym
import numpy as np
from collections import deque
from PIL import Image


class FlappyBirdWrapper(gym.Wrapper):
    """
    Wrapper pentru preprocesarea input-ului Flappy Bird:
    - conversie la grayscale
    - resize la 84x84
    - stack de 4 frames
    - normalizare la [0, 1]
    """
    
    def __init__(self, env, img_size=64, stack_frames=10):
        super().__init__(env)
        self.img_size = img_size
        self.stack_frames = stack_frames
        
        # deque pentru stocarea frame-urilor
        self.frames = deque(maxlen=stack_frames)
        
        # actualizează observation space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(stack_frames, img_size, img_size),
            dtype=np.float32
        )
    
    def _preprocess_frame(self, frame):
        """conversie RGB -> grayscale -> resize -> normalize"""
        # conversie la grayscale
        img = Image.fromarray(frame).convert('L')
        
        # resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # conversie la numpy și normalizare
        frame = np.array(img, dtype=np.float32) / 255.0
        
        return frame
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # preprocesează frame-ul inițial
        frame = self._preprocess_frame(obs)
        
        # umple stack-ul cu același frame
        for _ in range(self.stack_frames):
            self.frames.append(frame)
        
        return self._get_stacked_frames(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # preprocesează și adaugă noul frame
        frame = self._preprocess_frame(obs)
        self.frames.append(frame)
        
        # nu mai modificam reward-ul, îl lăsăm cum vine din mediu
        # mediul deja oferă:
        # +0.1 per frame
        # +1.0 pentru tub trecut
        # -1.0 la moarte
        # -0.5 la touch top
        
        return self._get_stacked_frames(), reward, terminated, truncated, info
    
    def _get_stacked_frames(self):
        """returnează stack-ul de frame-uri ca array numpy"""
        return np.stack(self.frames, axis=0)