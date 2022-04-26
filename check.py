import isaacgym
import torch  # We must import torch behind isaacgym
from elegantrl.envs.IsaacGym import IsaacVecEnv, IsaacOneEnv, check_isaac_gym


# Choose from one of the following:
# AllegroHand
# Ant  <--------
# Anymal
# AnymalTerrain
# BallBalance
# Cartpole
# FrankaCabinet
# Humanoid <----------
# Ingenuity
# Quadcopter
# ShadowHand  <----------
# Trifinger
check_isaac_gym('ShadowHand')
