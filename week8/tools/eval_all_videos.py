# Run lerobot-eval, but save a video for EVERY episode, not just the first 10.
# LeRobot hard-codes the limit (lerobot_eval.py: max_episodes_rendered = 10), so we
# wrap eval_policy_all and replace that argument with the number of episodes.
# Same flags as lerobot-eval. Policy behaviour is unchanged; only rendering differs.
import lerobot.scripts.lerobot_eval as lerobot_eval

_original = lerobot_eval.eval_policy_all


def eval_policy_all_every_video(*args, **kwargs):
    if kwargs.get("max_episodes_rendered", 0) > 0:      # 0 means recording mode: leave it alone
        kwargs["max_episodes_rendered"] = kwargs["n_episodes"]
    return _original(*args, **kwargs)


lerobot_eval.eval_policy_all = eval_policy_all_every_video

if __name__ == "__main__":
    lerobot_eval.main()
