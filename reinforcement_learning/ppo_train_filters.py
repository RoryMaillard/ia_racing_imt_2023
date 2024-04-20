import argparse
import uuid
import gym_donkeycar
import gym
from stable_baselines3 import PPO
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
        "imt-track-v0"
    ]

    parser = argparse.ArgumentParser(description="ppo_train")
    parser.add_argument(
        "--sim",
        type=str,
        default="/home/rory/Documents/IMT/PROCOMFINAL/DonkeySimLinux/donkey_sim.x86_64",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--test", action="store_true", help="load the trained model and play")
    parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
    parser.add_argument(
        "--env_name", type=str, default="imt-track-v0", help="name of donkey sim environment", choices=env_list
    )

    args = parser.parse_args()
    
    
    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)

    env_id = args.env_name


    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 100,
        "racer_name": "DDQN",
        "country": "USA",
        "bio": "Learning to drive w DDQN RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
    }

    if args.test:
        # Make an environment test our trained policy
        env = gym.make(args.env_name, conf=conf)

        model = PPO.load("ppo_donkey_filters")

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()

        print("done testing")

    else:

        # make gym env
        env = gym.make(args.env_name, conf=conf)
        env.observation_space = gym.spaces.Box(0, 255, (75, 160), np.uint8)

        #model = PPO.load("ppo_donkey_filters",env)
        model = PPO("MlpPolicy", env, verbose=1)
        for name,p in model.policy.mlp_extractor.named_parameters():
            print (name,p)
        nparams = np.sum([p.numel() for p in model.policy.features_extractor.parameters() if p.requires_grad])
        nparams2 = np.sum([p.numel() for p in model.policy.mlp_extractor.parameters() if p.requires_grad])
        print(nparams)
        print(nparams2)

        # set up model in learning mode with goal number of timesteps to complete
        model.learn(total_timesteps=10000)

        obs = env.reset()

        # Save the agent
        model.save("ppo_donkey_filters")
        print("done training")

    env.close()
