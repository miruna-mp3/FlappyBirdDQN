"""
Test play.py cu un model dummy (neantrenat)
"""
from dqn_agent import DQNAgent
from play import evaluate_agent

# Creează și salvează un model dummy
print("📦 Creare model dummy pentru test...")
agent = DQNAgent()
agent.save("dummy_model.pth")
print("   Model dummy salvat\n")

# Test evaluare (3 episoade, cu randare)
evaluate_agent(
    model_path="dummy_model.pth",
    n_episodes=3,
    render=True
)

print("\n✅ Test finalizat!")
print("   Scriptul play.py funcționează corect")
print("   După antrenare, folosește: python play.py")