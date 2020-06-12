#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:10:41 2020

@author: he
"""
def get_initial_state(self):
    """Returns initial RNN state for the current policy."""
    return [0]  # list of single state element (t=0)
                # you could also return multiple values, e.g., [0, "foo"]

def compute_actions(self,
                    obs_batch,
                    state_batches,
                    prev_action_batch=None,
                    prev_reward_batch=None,
                    info_batch=None,
                    episodes=None,
                    **kwargs):
    assert len(state_batches) == len(self.get_initial_state())
    new_state_batches = [[
        t + 1 for t in state_batches[0]
    ]]
    return ..., new_state_batches, {}

def learn_on_batch(self, samples):
    # can access array of the state elements at each timestep
    # or state_in_1, 2, etc. if there are multiple state elements
    assert "state_in_0" in samples.keys()
    assert "state_out_0" in samples.keys()
    
import tensorflow as tf

from ray.rllib.policy.sample_batch import SampleBatch

def policy_gradient_loss(policy, model, dist_class, train_batch):
    actions = train_batch[SampleBatch.ACTIONS]
    rewards = train_batch[SampleBatch.REWARDS]
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    return -tf.reduce_mean(action_dist.logp(actions) * rewards)


from ray.rllib.policy.tf_policy_template import build_tf_policy

# <class 'ray.rllib.policy.tf_policy_template.MyTFPolicy'>
MyTFPolicy = build_tf_policy(
    name="MyTFPolicy",
    loss_fn=policy_gradient_loss)

from ray.rllib.policy.tf_policy_template import build_tf_policy

# <class 'ray.rllib.policy.tf_policy_template.MyTFPolicy'>
MyTFPolicy = build_tf_policy(
    name="MyTFPolicy",
    loss_fn=policy_gradient_loss)
