from stable_baselines3.common.policies import ActorCriticPolicy


class CustomPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
    ):
        super(CustomPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

        # self.net_arch = {"pi": [128, 64, 32], "vf": [128, 64, 32]}
