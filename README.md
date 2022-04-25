Run the check program first.
```
"""
run the following code in bash before running.
export LD_LIBRARY_PATH=/xfs/home/podracer_steven/anaconda3/envs/rlgpu/lib
can't use os.environ['LD_LIBRARY_PATH'] = /xfs/home/podracer_steven/anaconda3/envs/rlgpu/lib
"""

import isaacgym
import torch  # We must import torch behind isaacgym
from elegantrl.envs.IsaacGym import IsaacVecEnv, IsaacOneEnv, check_isaac_gym

check_isaac_gym()
```

Then you can run the training python program.
```example/demo_isaacgym.py

if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU
    DRL_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ENV_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    demo_a2c_ppo(GPU_ID, DRL_ID, ENV_ID)
```
