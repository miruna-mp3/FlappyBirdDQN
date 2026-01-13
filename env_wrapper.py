import gymnasium as gym
import numpy as np
from collections import deque
from PIL import Image


class FlappyBirdWrapper(gym.Wrapper):
    """
    wrapper pentru preprocesarea pixelilor flappy bird
    """

    def __init__(self, env, img_size=84, stack_frames=4, frame_skip=2, use_binary=True):
        super().__init__(env)
        self.img_size = img_size
        self.stack_frames = stack_frames
        self.frame_skip = frame_skip
        self.use_binary = use_binary

        # stivă pentru ultimele frame-uri (informație temporală)
        self.frames = deque(maxlen=stack_frames)

        # definește spațiul de observare pentru gymnasium
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(stack_frames, img_size, img_size),
            dtype=np.float32
        )

    def _preprocess_frame(self, frame):
        # conversie la grayscale și resize la 84x84
        img = Image.fromarray(frame).convert('L')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        frame = np.array(img, dtype=np.float32)

        # binarizare - elimină fundalul și păstrează doar obiectele importante
        if self.use_binary:
            frame = np.where(frame < 150, 1.0, 0.0).astype(np.float32)
        else:
            frame = frame / 255.0

        return frame

    def reset(self, **kwargs):
        # resetează mediul și obține primul frame
        _, info = self.env.reset(**kwargs)
        rgb_frame = self.env.render()
        frame = self._preprocess_frame(rgb_frame)

        # umple stiva cu primul frame
        for _ in range(self.stack_frames):
            self.frames.append(frame)

        return self._get_stacked_frames(), info

    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False

        # frame skip - repetă acțiunea pentru mai multe frame-uri
        for _ in range(self.frame_skip):
            _, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        # preprocesează și adaugă noul frame în stivă
        rgb_frame = self.env.render()
        frame = self._preprocess_frame(rgb_frame)
        self.frames.append(frame)

        # reward shaping - penalizare pentru moarte, bonus pentru supraviețuire
        if terminated or truncated:
            total_reward = -1.0
        else:
            total_reward += 0.1

        # clipează reward-ul pentru stabilitate
        clipped_reward = np.clip(total_reward, -1.0, 1.0)
        return self._get_stacked_frames(), clipped_reward, terminated, truncated, info

    def _get_stacked_frames(self):
        # returnează stiva de frame-uri ca tensor numpy
        return np.stack(self.frames, axis=0)
