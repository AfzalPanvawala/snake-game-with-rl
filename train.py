import numpy as np
import matplotlib.pyplot as plt
import pygame
from game_env import SnakeEnv
from agent import DQNAgent

def run_snake():
    # Smaller 10x10 grid (100 total cells) so the snake can completely fill the board
    env = SnakeEnv(grid_size=10, use_distance_shaping=True, render_mode=True, fps=15)
    agent = DQNAgent(lr=0.001, gamma=0.95, use_double_dqn=True)

    state = env.reset()
    steps_per_food_history = []
    last_food_step = 0
    done = False

    print("RL Snake running! Goal: Fill all 100 cells of the 10x10 grid.")

    try:
        while not done:
            action = agent.get_action(state, eval_mode=False)
            next_state, reward, done, score = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state

            # Every apple eaten sharpens food-finding accuracy
            if reward in (10.0, 50.0):
                agent.epsilon = max(0.01, agent.epsilon * 0.90)
                agent.update_target_network()
                for _ in range(3):
                    agent.train_step()

                steps_taken = env.total_steps - last_food_step
                last_food_step = env.total_steps
                steps_per_food_history.append(steps_taken)
                accuracy_pct = (1.0 - agent.epsilon) * 100
                print(f"Food #{score} eaten! | Grid Filled: {len(env.snake)}/{env.max_cells} | Steps: {steps_taken} | Policy Accuracy: {accuracy_pct:.1f}%")

        print(f"\nGRID COMPLETELY FULL! Snake reached {len(env.snake)}/{env.max_cells} cells in {env.total_steps} steps!")
        pygame.time.wait(2500)  # Pause 2.5 seconds so you can see the completely full grid
        pygame.quit()

    except SystemExit:
        print("\nWindow closed manually.")

    finally:
        agent.save_checkpoint("snake_dqn.pth")
        if len(steps_per_food_history) > 1:
            moving_avg = [np.mean(steps_per_food_history[max(0, i-5):i+1]) for i in range(len(steps_per_food_history))]
            plt.figure(figsize=(10, 5))
            plt.plot(steps_per_food_history, label="Steps to Find Food", alpha=0.45, color="royalblue", marker="o")
            plt.plot(moving_avg, label="Moving Average Steps/Food (Lower = More Accurate)", color="darkorange", linewidth=2)
            plt.title("RL Snake (10x10 Grid): Food-Finding Accuracy Over Time")
            plt.xlabel("Food Items Eaten")
            plt.ylabel("Steps Taken to Reach Food")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.savefig("accuracy_curve.png", dpi=300)
            plt.show()

if __name__ == "__main__":
    run_snake()